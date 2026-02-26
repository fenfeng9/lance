// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::HashMap,
    fmt::Debug,
    pin::Pin,
    sync::{Arc, LazyLock, Mutex},
};

use arrow::array::AsArray;
use arrow_array::{Array, BinaryArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::RecordBatchStream;
use datafusion::physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream};
use datafusion_common::ScalarValue;
use deepsize::DeepSizeOf;
use futures::{stream::BoxStream, StreamExt, TryStream, TryStreamExt};
use lance_core::cache::LanceCache;
use lance_core::error::LanceOptionExt;
use lance_core::utils::mask::{NullableRowAddrSet, RowAddrTreeMap, RowSetOps};
use lance_core::{Error, Result, ROW_ID};
use roaring::RoaringBitmap;
use snafu::location;
use tracing::instrument;

use super::{bitmap::BitmapIndex, AnyQuery, IndexStore, LabelListQuery, ScalarIndex};
use super::{BuiltinIndexType, SargableQuery, ScalarIndexParams};
use super::{MetricsCollector, SearchResult};
use crate::frag_reuse::FragReuseIndex;
use crate::pbold;
use crate::scalar::bitmap::BitmapIndexPlugin;
use crate::scalar::expression::{LabelListQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    DefaultTrainingRequest, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
    VALUE_COLUMN_NAME,
};
use crate::scalar::{CreatedIndex, UpdateCriteria};
use crate::{Index, IndexType};

pub const BITMAP_LOOKUP_NAME: &str = "bitmap_page_lookup.lance";
const LABEL_LIST_NULLS_NAME: &str = "label_list_nulls.lance";
pub const LABEL_LIST_NULLS_MIN_VERSION: i32 = 1;
const LABEL_LIST_INDEX_VERSION: u32 = 1;

static LABEL_LIST_NULLS_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![Field::new(
        "nulls",
        DataType::Binary,
        false,
    )]))
});

#[async_trait]
trait LabelListSubIndex: ScalarIndex + DeepSizeOf {
    async fn search_exact(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<NullableRowAddrSet> {
        let result = self.search(query, metrics).await?;
        match result {
            SearchResult::Exact(row_ids) => {
                // Label list semantics treat NULL elements as non-matches, so only TRUE/FALSE
                // results should remain for array_has_any/array_has_all when the list itself
                // is non-NULL. Clear nulls to avoid propagating element-level NULLs.
                Ok(row_ids.with_nulls(RowAddrTreeMap::new()))
            }
            _ => Err(Error::Internal {
                message: "Label list sub-index should return exact results".to_string(),
                location: location!(),
            }),
        }
    }
}

impl<T: ScalarIndex + DeepSizeOf> LabelListSubIndex for T {}

/// A scalar index that can be used on `List<T>` columns to
/// accelerate list membership filters such as `array_has_all`, `array_has_any`,
/// and `array_has` / `array_contains`, using an underlying bitmap index.
#[derive(Clone, Debug, DeepSizeOf)]
pub struct LabelListIndex {
    values_index: Arc<dyn LabelListSubIndex>,
    list_nulls: Arc<RowAddrTreeMap>,
}

impl LabelListIndex {
    fn new(values_index: Arc<dyn LabelListSubIndex>, list_nulls: Arc<RowAddrTreeMap>) -> Self {
        Self {
            values_index,
            list_nulls,
        }
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let values_index =
            BitmapIndex::load(store.clone(), frag_reuse_index.clone(), index_cache).await?;
        let list_nulls = read_list_nulls(store, frag_reuse_index).await?;
        Ok(Arc::new(Self::new(values_index, Arc::new(list_nulls))))
    }
}

#[async_trait]
impl Index for LabelListIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "LabeListIndex is not a vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        self.values_index.prewarm().await
    }

    fn index_type(&self) -> IndexType {
        IndexType::LabelList
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        self.values_index.statistics()
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

impl LabelListIndex {
    fn search_values<'a>(
        &'a self,
        values: &'a Vec<ScalarValue>,
        metrics: &'a dyn MetricsCollector,
    ) -> BoxStream<'a, Result<NullableRowAddrSet>> {
        futures::stream::iter(values)
            .then(move |value| {
                let value_query = SargableQuery::Equals(value.clone());
                async move { self.values_index.search_exact(&value_query, metrics).await }
            })
            .boxed()
    }

    async fn set_union<'a>(
        &'a self,
        mut sets: impl TryStream<Ok = NullableRowAddrSet, Error = Error> + 'a + Unpin,
        single_set: bool,
    ) -> Result<NullableRowAddrSet> {
        let mut union_bitmap = sets.try_next().await?.unwrap();
        if single_set {
            return Ok(union_bitmap);
        }
        while let Some(next) = sets.try_next().await? {
            union_bitmap |= &next;
        }
        Ok(union_bitmap)
    }

    async fn set_intersection<'a>(
        &'a self,
        mut sets: impl TryStream<Ok = NullableRowAddrSet, Error = Error> + 'a + Unpin,
        single_set: bool,
    ) -> Result<NullableRowAddrSet> {
        let mut intersect_bitmap = sets.try_next().await?.unwrap();
        if single_set {
            return Ok(intersect_bitmap);
        }
        while let Some(next) = sets.try_next().await? {
            intersect_bitmap &= &next;
        }
        Ok(intersect_bitmap)
    }
}

#[async_trait]
impl ScalarIndex for LabelListIndex {
    #[instrument(skip_all, level = "debug")]
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<LabelListQuery>().unwrap();

        let row_ids = match query {
            LabelListQuery::HasAllLabels(labels) => {
                let values_results = self.search_values(labels, metrics);
                self.set_intersection(values_results, labels.len() == 1)
                    .await
            }
            LabelListQuery::HasAnyLabel(labels) => {
                let values_results = self.search_values(labels, metrics);
                self.set_union(values_results, labels.len() == 1).await
            }
        }?;
        let row_ids = if self.list_nulls.as_ref().is_empty() {
            row_ids
        } else {
            let mut nulls = row_ids.null_rows().clone();
            nulls |= self.list_nulls.as_ref();
            row_ids.with_nulls(nulls)
        };
        Ok(SearchResult::Exact(row_ids))
    }

    fn can_remap(&self) -> bool {
        true
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        self.values_index.remap(mapping, dest_store).await?;
        let remapped_nulls =
            RowAddrTreeMap::from_iter(self.list_nulls.row_addrs().unwrap().filter_map(|addr| {
                let addr_as_u64 = u64::from(addr);
                mapping
                    .get(&addr_as_u64)
                    .copied()
                    .unwrap_or(Some(addr_as_u64))
            }));
        write_list_nulls(dest_store, remapped_nulls).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
        })
    }

    /// Add the new data into the index, creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        valid_old_fragments: Option<&RoaringBitmap>,
    ) -> Result<CreatedIndex> {
        let list_nulls = Arc::new(Mutex::new(RowAddrTreeMap::new()));
        let new_data = track_list_nulls(new_data, list_nulls.clone());
        self.values_index
            .update(unnest_chunks(new_data)?, dest_store, valid_old_fragments)
            .await?;
        let mut merged_nulls = (*self.list_nulls).clone();
        let new_nulls = list_nulls.lock().unwrap().clone();
        if !new_nulls.is_empty() {
            merged_nulls |= &new_nulls;
        }
        write_list_nulls(dest_store, merged_nulls).await?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList))
    }
}

fn extract_flatten_indices(list_arr: &dyn Array) -> UInt64Array {
    if let Some(list_arr) = list_arr.as_list_opt::<i32>() {
        let mut indices = Vec::with_capacity(list_arr.values().len());
        let offsets = list_arr.value_offsets();
        for (offset_idx, w) in offsets.windows(2).enumerate() {
            let size = (w[1] - w[0]) as u64;
            indices.extend((0..size).map(|_| offset_idx as u64));
        }
        UInt64Array::from(indices)
    } else if let Some(list_arr) = list_arr.as_list_opt::<i64>() {
        let mut indices = Vec::with_capacity(list_arr.values().len());
        let offsets = list_arr.value_offsets();
        for (offset_idx, w) in offsets.windows(2).enumerate() {
            let size = (w[1] - w[0]) as u64;
            indices.extend((0..size).map(|_| offset_idx as u64));
        }
        UInt64Array::from(indices)
    } else {
        unreachable!(
            "Should verify that the first column is a list earlier. Got array of type: {}",
            list_arr.data_type()
        )
    }
}

/// Collect row_ids for list-level NULLs before unnest; unnest drops NULL lists entirely.
fn track_list_nulls(
    source: SendableRecordBatchStream,
    list_nulls: Arc<Mutex<RowAddrTreeMap>>,
) -> SendableRecordBatchStream {
    let schema = source.schema();
    let stream = source.try_filter_map(move |batch| {
        let list_nulls = list_nulls.clone();
        async move {
            record_list_nulls(&batch, &list_nulls)?;
            Ok(Some(batch))
        }
    });

    Box::pin(RecordBatchStreamAdapter::new(schema, stream))
}

fn record_list_nulls(
    batch: &RecordBatch,
    list_nulls: &Arc<Mutex<RowAddrTreeMap>>,
) -> datafusion_common::Result<()> {
    let values = batch.column_by_name(VALUE_COLUMN_NAME).expect_ok()?;
    let row_ids = batch.column_by_name(ROW_ID).expect_ok()?;
    let row_ids = row_ids.as_any().downcast_ref::<UInt64Array>().unwrap();

    let mut local_nulls = RowAddrTreeMap::new();
    for i in 0..values.len() {
        if values.is_null(i) {
            local_nulls.insert(row_ids.value(i));
        }
    }
    if !local_nulls.is_empty() {
        let mut guard = list_nulls.lock().unwrap();
        *guard |= &local_nulls;
    }
    Ok(())
}

fn unnest_schema(schema: &Schema) -> SchemaRef {
    let mut fields_iter = schema.fields.iter().cloned();
    let key_field = fields_iter.next().unwrap();
    let remaining_fields = fields_iter.collect::<Vec<_>>();

    let new_key_field = match key_field.data_type() {
        DataType::List(item_field) | DataType::LargeList(item_field) => Field::new(
            key_field.name(),
            item_field.data_type().clone(),
            item_field.is_nullable() || key_field.is_nullable(),
        ),
        other_type => {
            unreachable!(
                "The first field in the schema must be a List or LargeList type. \
                Found: {}. This should have been verified earlier in the code.",
                other_type
            )
        }
    };

    let all_fields = vec![Arc::new(new_key_field)]
        .into_iter()
        .chain(remaining_fields)
        .collect::<Vec<_>>();

    Arc::new(Schema::new(Fields::from(all_fields)))
}

fn unnest_batch(
    batch: arrow::record_batch::RecordBatch,
    unnest_schema: SchemaRef,
) -> datafusion_common::Result<RecordBatch> {
    let mut columns_iter = batch.columns().iter().cloned();
    let key_col = columns_iter.next().unwrap();
    let remaining_cols = columns_iter.collect::<Vec<_>>();

    let remaining_fields = unnest_schema
        .fields
        .iter()
        .skip(1)
        .cloned()
        .collect::<Vec<_>>();

    let remaining_batch = RecordBatch::try_new(
        Arc::new(Schema::new(Fields::from(remaining_fields))),
        remaining_cols,
    )?;

    let flatten_indices = extract_flatten_indices(key_col.as_ref());

    let flattened_remaining =
        arrow_select::take::take_record_batch(&remaining_batch, &flatten_indices)?;

    let new_key_values = if let Some(key_list) = key_col.as_list_opt::<i32>() {
        let value_start = key_list.value_offsets()[key_list.offset()] as usize;
        let value_stop = key_list.value_offsets()[key_list.len()] as usize;
        key_list
            .values()
            .slice(value_start, value_stop - value_start)
            .clone()
    } else if let Some(key_list) = key_col.as_list_opt::<i64>() {
        let value_start = key_list.value_offsets()[key_list.offset()] as usize;
        let value_stop = key_list.value_offsets()[key_list.len()] as usize;
        key_list
            .values()
            .slice(value_start, value_stop - value_start)
            .clone()
    } else {
        unreachable!("Should verify that the first column is a list earlier")
    };

    let all_columns = vec![new_key_values]
        .into_iter()
        .chain(flattened_remaining.columns().iter().cloned())
        .collect::<Vec<_>>();

    datafusion_common::Result::Ok(arrow::record_batch::RecordBatch::try_new(
        unnest_schema,
        all_columns,
    )?)
}

fn unnest_chunks(
    source: Pin<Box<dyn RecordBatchStream + Send>>,
) -> Result<SendableRecordBatchStream> {
    let unnest_schema = unnest_schema(source.schema().as_ref());
    let unnest_schema_copy = unnest_schema.clone();
    let source = source.try_filter_map(move |batch| {
        std::future::ready(Some(unnest_batch(batch, unnest_schema.clone())).transpose())
    });

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        unnest_schema_copy,
        source,
    )))
}

async fn read_list_nulls(
    store: Arc<dyn IndexStore>,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
) -> Result<RowAddrTreeMap> {
    let reader = match store.open_index_file(LABEL_LIST_NULLS_NAME).await {
        Ok(reader) => reader,
        Err(err) => {
            // Old LabelList indices don't have label_list_nulls.lance; treat as empty for compatibility.
            let is_not_found = matches!(
                &err,
                Error::IO { source, .. }
                    if source
                        .downcast_ref::<object_store::Error>()
                        .map(|err| matches!(err, object_store::Error::NotFound { .. }))
                        .unwrap_or(false)
            );
            if is_not_found {
                return Ok(RowAddrTreeMap::default());
            }
            return Err(err);
        }
    };

    let batch = reader.read_range(0..1, None).await?;
    let null_map = match batch.num_rows() {
        0 => RowAddrTreeMap::default(),
        1 => {
            let bytes = batch
                .column(0)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(0);
            RowAddrTreeMap::deserialize_from(bytes)?
        }
        _ => {
            return Err(Error::Internal {
                message: "label_list_nulls.lance should contain at most one row".to_string(),
                location: location!(),
            });
        }
    };

    if let Some(frag_reuse_index) = frag_reuse_index {
        Ok(frag_reuse_index.remap_row_addrs_tree_map(&null_map))
    } else {
        Ok(null_map)
    }
}

async fn write_list_nulls(store: &dyn IndexStore, null_map: RowAddrTreeMap) -> Result<()> {
    let mut writer = store
        .new_index_file(LABEL_LIST_NULLS_NAME, LABEL_LIST_NULLS_SCHEMA.clone())
        .await?;
    let mut bytes = Vec::new();
    null_map.serialize_into(&mut bytes)?;
    let batch = RecordBatch::try_new(
        LABEL_LIST_NULLS_SCHEMA.clone(),
        vec![Arc::new(BinaryArray::from_vec(vec![&bytes]))],
    )?;

    writer.write_record_batch(batch).await?;
    writer.finish().await
}

#[derive(Debug, Default)]
pub struct LabelListIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for LabelListIndexPlugin {
    fn name(&self) -> &str {
        "LabelList"
    }

    fn new_training_request(
        &self,
        _params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if !matches!(
            field.data_type(),
            DataType::List(_) | DataType::LargeList(_)
        ) {
            return Err(Error::InvalidInput {
                source: format!(
                    "LabelList index can only be created on List or LargeList type columns. Column has type {:?}",
                    field.data_type()
                )
                .into(),
                location: location!(),
            });
        }

        Ok(Box::new(DefaultTrainingRequest::new(
            TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        )))
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        LABEL_LIST_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(LabelListQueryParser::new(index_name)))
    }

    /// Train a new index
    ///
    /// The provided data must fulfill all the criteria returned by `training_criteria`
    /// and the plugin can rely on this fact.
    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        if fragment_ids.is_some() {
            return Err(Error::InvalidInput {
                source: "LabelList index does not support fragment training".into(),
                location: location!(),
            });
        }

        let schema = data.schema();
        let field = schema
            .column_with_name(VALUE_COLUMN_NAME)
            .ok_or_else(|| Error::InvalidInput {
                source: "Index training data missing value column"
                    .to_string()
                    .into(),
                location: location!(),
            })?
            .1;

        if !matches!(
            field.data_type(),
            DataType::List(_) | DataType::LargeList(_)
        ) {
            return Err(Error::InvalidInput {
                source: format!(
                    "LabelList index can only be created on List or LargeList type columns. Column has type {:?}",
                    field.data_type()
                )
                .into(),
                location: location!(),
            });
        }

        let list_nulls = Arc::new(Mutex::new(RowAddrTreeMap::new()));
        let data = track_list_nulls(data, list_nulls.clone());
        let data = unnest_chunks(data)?;
        let bitmap_plugin = BitmapIndexPlugin;
        bitmap_plugin
            .train_index(data, index_store, request, fragment_ids, progress)
            .await?;
        let list_nulls = list_nulls.lock().unwrap().clone();
        write_list_nulls(index_store, list_nulls).await?;
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&pbold::LabelListIndexDetails::default())
                .unwrap(),
            index_version: LABEL_LIST_INDEX_VERSION,
        })
    }

    /// Load an index from storage
    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(
            LabelListIndex::load(index_store, frag_reuse_index, cache).await?
                as Arc<dyn ScalarIndex>,
        )
    }
}
