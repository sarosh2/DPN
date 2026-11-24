# Dataset configs

Each `configs/<dataset>/dataset.json` is the **only required input config**. It specifies the data source, exact split, preprocessing, original maximal singleton DPN, original outer pruning, reference MLP shape, regularization and training details.

At run time, `configs/<dataset>/<seed>/` receives generated `<model>.config` JSON files and the verified accepted pruning ticket. Existing configs are immutable until archived. The resolver creates an MLP config before a Block DPN config. Example: `python run.py --datasets jannis --seeds 42 --models block_matched_dpn`.

The one-time historical importer creates `configs/<dataset>/<seed>/*.config`
for all seven model families, taking exact geometry from frozen contracts when
available. It uses corrected Step 89 regularized Block DPNs. The release already includes the 20 accepted tickets and 140 frozen model
configs; no migration is needed to run the five-model subset.

## Automatic dataset acquisition

The built-in configs include an optional `data_acquisition` section. It
specifies the official archive URL, dataset subdirectory, pinned archive
SHA256, and seven pinned file SHA256 values. This section is excluded from
the legacy scientific configuration fingerprint so that adding verified pins
or changing an archive mirror does not invalidate the 20 migrated tickets or
140 model configs. The loader enforces the pins; the downloaded **data bytes**
remain in the per-split SHA256 manifest and checkpoint identity. Do not change
the pins to substitute a different dataset while reusing old tickets.

No extra script is needed: the main runner calls `src/data_io.py` on first
load. New datasets can use a source with an exact archive checksum and
official prepared splits, or add a compatible acquisition adapter to the
existing `src/data_io.py` if the source has a new storage format.
