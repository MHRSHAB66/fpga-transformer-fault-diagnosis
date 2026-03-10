# Day 1

## Completed
- Repository created and pushed
- Initial project structure created
- `config.py` added
- Data loading pipeline initialized
- Data inspection script implemented
- Synthetic MAFAULDA-like dataset generated for development
- Windowing pipeline implemented
- Initial dataset tensors prepared

## Known facts
- Target dataset is MAFAULDA
- Files are CSV
- Each file has 8 columns
- Current pipeline expects 4 target classes:
  - normal
  - imbalance
  - misalignment
  - bearing_fault
- Data is converted into windows of shape `(128, 8)`
- Current synthetic dataset produced:
  - `X shape = (32, 128, 8)`
  - `y shape = (32,)`

## Still unknown
- Exact folder naming convention of the real MAFAULDA dataset
- Exact real column names in the original dataset
- Exact size and structure of the full downloaded dataset
- Exact FPGA board
- Exact real sensor for deployment

## Notes
- A synthetic dataset was used to validate the pipeline before downloading the full dataset
- The data pipeline is now ready to be tested on the real dataset