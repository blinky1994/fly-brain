from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy import sparse

root = Path(__file__).resolve().parent
raw = root / "data/raw"
out = root / "data/processed"
out.mkdir(parents=True, exist_ok=True)

annotations = feather.read_table(raw / "annotations.feather").to_pandas()
keep = (
    annotations["superclass"].fillna("").str.strip().ne("")
    & ~annotations["status"].fillna("").str.contains(
        "glia", case=False, regex=False
    )
)
neurons = annotations.loc[keep].sort_values("bodyId").reset_index(drop=True)
ids = pd.Index(neurons["bodyId"])

assert ids.is_unique, "Duplicate neuron IDs"
assert len(ids) == 166700, "Unexpected neuron selection"

nt = feather.read_table(raw / "neurotransmitters.feather").to_pandas()
assert nt["body"].is_unique, "Duplicate neurotransmitter records"
neurons["consensus_nt"] = neurons["bodyId"].map(
    nt.set_index("body")["consensus_nt"]
)

types = neurons["type"].fillna("")
is_kc = types.str.startswith("KC").to_numpy()
is_mbon = types.eq("MBON11").to_numpy()

pres, posts, counts = [], [], []
raw_rows = 0

print("Reading connections...", flush=True)
with pa.memory_map(str(raw / "edges.feather"), "r") as source:
    reader = ipc.open_file(source)

    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)

        def column(name):
            return batch.column(
                batch.schema.get_field_index(name)
            ).to_numpy(zero_copy_only=False)

        pre = ids.get_indexer(column("body_pre"))
        post = ids.get_indexer(column("body_post"))
        weight = column("weight")
        assert np.all(weight > 0), "Non-positive synapse count"

        valid = (pre >= 0) & (post >= 0)
        pres.append(pre[valid].astype(np.int32))
        posts.append(post[valid].astype(np.int32))
        counts.append(weight[valid].astype(np.int64))
        raw_rows += len(weight)

pre = np.concatenate(pres)
post = np.concatenate(posts)
weight = np.concatenate(counts)
del pres, posts, counts

# Rows receive signals; columns send signals.
# These are anatomical synapse counts, not yet physiological strengths.
matrix = sparse.coo_matrix(
    (weight, (post, pre)), shape=(len(ids), len(ids))
).tocsr()
matrix.sum_duplicates()
matrix.sort_indices()

assert int(matrix.sum()) == int(weight.sum())

# Mark EXISTING Kenyon-cell -> MBON11 connections.
# No new neural connections are introduced.
plastic = is_kc[pre] & is_mbon[post]
np.savez_compressed(
    out / "learning_connections.npz",
    pre=pre[plastic],
    post=post[plastic],
    synapse_count=weight[plastic],
)

print("Saving full graph...", flush=True)
sparse.save_npz(out / "synapse_counts.npz", matrix, compressed=False)
feather.write_feather(neurons, out / "neurons.feather")

report = {
    "dataset": "MaleCNS v1.0",
    "neurons": len(ids),
    "source_edge_rows": raw_rows,
    "retained_edge_rows": len(weight),
    "excluded_edge_rows": raw_rows - len(weight),
    "unique_directed_connections": matrix.nnz,
    "retained_synaptic_contacts": int(weight.sum()),
    "KC_to_MBON11_edge_rows": int(plastic.sum()),
    "missing_transmitter_labels": int(neurons["consensus_nt"].isna().sum()),
    "matrix_orientation": "rows=postsynaptic, columns=presynaptic",
    "learning_run": False,
}
(out / "build_report.json").write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))
print("\nNeurotransmitter labels:")
print(neurons["consensus_nt"].fillna("MISSING").value_counts().to_string())
print("\nSaved in:", out)
