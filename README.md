## Instructions

### Preprocess Data
```
cd data
python prepare_data.py googleplaystore.csv
```

### Train Model
```
nohup python -u train.py --root_path=model_v0 > log/v0.log 2>&1 &
```

### Start Server
```
nohup python -u server.py --batch_size=1 --use_beam_search=True --root_path=model_v0 > log/1-24.server.log 2>&1 &
```
