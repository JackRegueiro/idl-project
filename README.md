# IDL Project
## Distorted Embedding Safety (DES) Implementation
### Setup
1. Create a (Conda) Environment
```bash
conda create -n idl-project python=3.13.2
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the code via the `run.ipynb` notebook. Put implementations in the appropriate classes.

---

### Instructions for Running the Data Preprocessing Pipeline

#### 1. Dataset Loading and Validation
```bash
# From the idl-project directory:
python data/test_dataset.py 
```
This verifies the dataset is loaded correctly and filters to the sexual category, which contains the 6,911 prompt pairs used in the DES paper.

#### 2. Exploratory Data Analysis (Optional)
```bash
# From the idl-project directory:
cd data
python data_eda.py 
# Remember to go back to the root directory for the next steps!
cd ..
```
This generates visualizations of the dataset distribution, concept frequencies, and prompt lengths.

#### 3. Generate Embeddings
```bash
# From the idl-project directory:
python data/load_clip_encoder.py 
```
This creates four embedding files in `./data/embeddings/`:
- `safe_embeddings.pt`
- `unsafe_embeddings.pt`
- `nudity_embedding.pt`
- `unconditioned_embedding.pt`

#### 4. Generate Target Vectors
```bash
# From the idl-project directory:
python data/generate_target_vectors.py 
```
This completes the pipeline by:
- Loading the embeddings from step 3
- Applying Algorithm 1 from the DES paper
- Saving the target vectors to `./data/target_vectors/`

#### 5. Update Dataset with Target Vectors
```bash
# From the idl-project directory:
python data/generate_target_vectors.py --embeddings_dir "embeddings" --output_dir "data/target_vectors" --scaling_factor 200.0 --update_dataset --data_path "CoPro Dataset/CoPro_v1.0.json"
```
Remember, initially we created the dataset in dataset.py with init function create a tuple of 3 items, (None, unsafe, safe). 


This step replaces the `None` values in the dataset tuples with the generated target vectors. When the `--update_dataset` flag is passed to `generate_target_vectors.py`, the script:
- Loads the dataset from the specified path
- Replaces each `None` in the first position with the corresponding target vector
- The data structure becomes: `(target_vector, unsafe_prompt, safe_prompt)`

### Additional Notes
- Ensure that the paths provided in the commands are correct and that the necessary files exist in those locations.
- If you encounter any errors, double-check the paths and ensure that all previous steps have been completed successfully.
- Make sure to handle any potential errors that may arise during execution, such as missing files or incorrect paths.
- 
---