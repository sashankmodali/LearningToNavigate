# LearningToNavigate

#### Preparing conda env
Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
```bash
# We require python>=3.7 and cmake>=3.10
conda create -n habitat python=3.7 cmake=3.14.0
conda activate habitat
```

#### conda install habitat-sim
- To install habitat-sim with bullet physics
   ```
   conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
   ```

#### Clone the repository
```
git clone https://github.com/sashankmodali/LearningToNavigate.git
cd LearningToNavigate
```



Install Habitat-lab using following commands: 
```
git clone  https://github.com/sashankmodali/habitat-lab.git
cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all # install habitat and habitat_baselines
```


#### Datasets
Dataset can be downloaded from [here](https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip) </br>
Gibson Scene datasets can be downloaded from [here](https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform)
</br>
Object datasets can be downloaded from [here](http://dl.fbaipublicfiles.com/habitat/objects_v0.2.zip)</br>

#### Folder Structure
Folder structure should be as follows:</br>
```
LearningToNavigate/
  habitat-lab/
  data/
    scene_datasets/
      gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        gibson/
          v1/
            train/
            val/
            ...
    object_datasets/
      banana.glb
      ...        
```

#### Create symbolic link of data folder inside Habitat Lab

```
cd habitat-lab
ln -s ../data data
cd ..
```

#### After setting up the environment:

1. For Milestone 1, run the following:
```
. milestone1.sh
```
  OR
```
conda activate habitat

python main.py --print_images 1
```
2. To generate training data and train depth1, run the following:
```
. generate_training_data.sh

python train_depth1.py
```
  OR
```
conda activate habitat

python main.py --print_images 1 -d ./training_data/ -el 10000 --task generate_train

python train_depth1.py
```
3. For Milestone 2 , run the following:
```
. milestone2.sh
```
  OR
```
conda activate habitat

python nslam.py --split val --eval 1 --train_global 0 --train_local 0 --train_slam 0 --load_global pretrained_models/model_best.global --load_local pretrained_models/model_best.local --load_slam pretrained_models/model_best.slam -n 1 --print_images 1

python generate_video.py
```
4. For Final Evaluations , run the following:
```
. eval_ppo_st.sh
```
  AND
```
. eval_ppo.sh
```
  AND
```
. eval_ans.sh
```
Then, the results can be obtained in /tmp/dump/[experiment]/episodes/1/1/
#### After replacing tmp directory line in generate_video.py
To generate video, run, 
```
python generate_video.py
```
Then, the results can be obtained in /tmp/dump/[experiment]/episodes/1/1video.avi
