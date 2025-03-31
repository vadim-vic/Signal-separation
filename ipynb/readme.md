# Notebooks to illustrate the signal separation problem
This list of Google Colab notebooks loads two data files. Take a look at the instructions below on how to load. The number before each title is the number that precedes the filename. 

## Data-related files

- Visualize the dimensionality of the span of signal basis, [10](10_SingularValuesDecomposition.ipynb)
- Reconstruct the mixed signal given a known basis, [11](11_GetData_FindTheBasis.ipynb)
- Select features and their optimal number to approximate the target, [12](12_SingularValuesDecomposition.ipynb)
- Find the centroids of the signal clusters, [9](9_Distance_to_6bit.ipynb)
- Two-class Aloha collision detector, early draft, [ACD2](AlohaCollisionDetector2class_Feb7.ipynb)
- Four-class Aloha collision detector, early draft, [ACD](AlohaCollisionDetector.ipynb)
- Three approaches to the signal processing, regression, and SSA: [1](1_SimpleRegression.ipynb), [5](5_SSAfeatures1series.ipynb), [8](8_PipelineToFeatureCollection.ipynb)
- Dictionary of models to approximate the mixed signal, [14](14_Base_Exhaustive_Search.ipynb)
- Test the reconstruction for various noise levels and number of mixed signals [13](13_Exhaustive_Base.ipynb)
- For two time series, use a modification of the Hausdorff metric [15](15_Hausdorf_distance_clustering.ipynb)

## Examples and plots
- An example of how to collect indices of the cartesian product of sets [16](16_Example_Cartesian_UpToC.ipynb)
- Plot the probability of the collision-free transmission, [1](1_Plot_Birthday_Probability_NQ.ipynb)
- An example of the shift function in self-modeling, [93](93_Example_shift.ipynb)
- An example of Singular Spectrum Analysis to plot two PC, [6](7_MixAndPlot_SSA.ipynb)
- Operations with complex vectors sometimes visually differ, [97](96_Example_ComplexVectorProjection.ipynb)  
- How to use a separate function .py file in the Colab notebook, [.ipynb](example_utility.ipynb) and [.py](example_utility.py)

## The Colad data and function loading 
To run the data-related Google Colab notebooks, two files from the [data folder](../data/) should be uploaded to the notebook drive. After you open a notebook in Colab, press the “Files” icon on the left panel of Colab (the fifth from the top, below the key). Then, press the “Upload to session storage” icon right below the word “Files”. From your local disk, upload the files inphase_quadrature_data.json, inphase_quadrature_noise.json, and inphase_quadrature_lib.npy. The last one is attached along with this text. In the Colab menu, click “Runtime” and select the item “Run all”. After the first cell runs, Colab
asks the access to the uploaded files. Press the button “Continue” each time to let Google Colab access your Google Drive, there will be several consequent requests. The experiment runs until the end. The figure shows the orange “Files” icon and the “Runtime” menu open.
<!--![Upload the files to Google Colab Python notebook to run the computational
experiment.](../latex/fig_demo_upload.png)-->
<img src="https://github.com/vadim-vic/Signal-separation/blob/main/latex/fig_demo_upload.png?raw=true)" alt="Upload the files to Google Colab Python notebook to run the computational
experiment." width="400" height="200">


<!-- OLD README Feb 7th, 2025
# The Independent Component Analysis is used for signal separation, the challenge is the single signal receiver  

1. [The Aloha RFID collision detection classifier model description](latex/CollisionDetector.pdf), Feb 7
2. [Two-class Aloha collision detection with RBF and Logistic Regression](ipynb/AlohaCollisionDetector2class_Feb7.ipynb), Feb 7
3. [Plot the probability of birthdays' collision](ipynb/1_Plot_Birthday_Probability_NQ.ipynb) 
4. [Find the clusters and their centroids in the signal collection](/ipynb/9_Distance_to_6bit.ipynb), Feb 2
5. [Analyze the dimensionality of the span of basis signals](/ipynb/10_SingularValuesDecomposition.ipynb), Feb 9
6. [New data generation procedure to reconstruct the mixed signals](/ipynb/11_GetData_FindTheBasis.ipynb), Feb 19
   
## Examples
1. Import functions from files in the Google Disc to the Google Colab: [example_utility.ipynb](examples/example_utility.ipynb), [example_utility.py](examples/example_utility.py)
2. [Collect indices of the cartesian product of 1, ..., C sets](examples/16_Example_Cartesian_UpToC.ipynb)
-->
