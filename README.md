# HumanActivityRecognition
 
 
I. 	Datasets, Pre-processing and Classes- 
   We originally decided to take 10 activities and started using those only, but because it increases weights a lot and uploading it on server while deploying takes up more storage, it charged us more. So, we now limit using to 6 activities. These 6 classes we used of UCF-101 Dataset are– 
•	Baby Crawling 
•	Typing 
•	Knitting 
•	Walking a dog 
•	Surfing 
•	Playing Violin 
 
1. Video Data Set Preparation- 
We have stored video in UCF-101 file in their corresponding classes. There is a test list and train list prepared to separate the videos into their respective classes. First, we created tags of each video to know which label it belonged to. 
 
2. Image Data Set Preparation along with classes- 
Now from every, video we take out frames at a particular rate. 
 
3. Image processing on the frames- 
We have tried many image processing filters and applied these three. 
Now we retrieve each frame and apply Gamma correction to increase the brightness of the image and then we have sharpened 2 times to remove the blurriness of the image and then applied median filter to remove the noise and then save it to another folder. We have used PIL for image sharpening and median filter and CV2 for gamma correction. So, we have converted the image from CV2 to PIL format by changing the colour by BGR2RGB. 
 
We have tried many image processing filters and used three filters – Median filter, Gamma Correction filter and Sharpening the image. These filters helped us in improving the quality of the frames that we have got from the videos. The improvement in the quality of images helped us to increase the overall efficiency of the program as well as the accuracy also got increased. We have applied these techniques in both the training dataset and testing dataset. 
 
The process involved in improving the efficiency of the image is as follows- 
 
▪	We retrieve each frame from the train_4 folder one by one and apply Gamma correction. Gamma Correction is a filter that controls the overall brightness of an image. Images which are not properly corrected can look either bleached out, or too dark. Varying the amount of gamma correction changes not only the brightness, but also the ratios of red to green to blue. We have applied this to increase the brightness of the image that have dark background and to enhance the foreground to extract more features for the model. 
 
▪	After that we take each frame that has been gamma corrected and sharpen the image. Sharpening is an image-manipulation technique for making the outlines of a digital image look more distinct. Sharpening increases the contrast between edge pixels and emphasizes the transition between dark and light areas. Sharpening increases local contrast and brings out fine detail. We have applied this two times to increase the overall contrast between edge pixels. This also helps in extracting different movements of the activities performed that are difficult to capture for the model to capture otherwise.   
 
 
▪	After sharpening the image 2 times we have applied the median filter. The median filter is normally used to reduce noise in an image, somewhat like the mean filter. However, it often does a better job than the mean filter of preserving useful detail in the image. Such noise reduction is a typical pre-processing step to improve the results of later processing such as to extract features for the models. 
 
These all techniques improve the overall quality of the images and to extract more features for the model to get trained better and to increase the accuracy of the model. After applying all these filters, we save these frames back to train_6 folder. We have used these techniques three times in our code first for the training dataset for which we have taken the frames from train_4 and saved back to train_6, accuracy calculation for which we have taken the frames form temp_4 and saved back to temp_5. Similarly, we have used these techniques for predicting a given activity. 
 
We have used both PIL and CV2 modules for image processing techniques. Image sharpening is done by PIL module and same for the median filter. The gamma correction is done using CV2 module. For converting an image to be used in CV2 module to PIL module we have a function that changes the colour from BGR2RGB. 
 
 
A limitation we face while applying image processing is that because we have so many classes of image as opposed to having only 2 classes, we had to choose only those processing techniques which on an average enhances all the images. Like Baby crawling is still a bit noisy, but typing is way better and surfing is also sharpened. 
 	 
4. Labels stored- 
After applying image processing, we are storing the name of every image along with its label in a csv folder. 
 
5. Training dataset prepared- 
Now, we first load all the images using glob library and then we reshape them to 224*224 size. After that we are changing the image in a NumPy array.  
Then we are splitting our whole training dataset into two parts: train and validation sets. The distribution is random but normalised using stratify. Thus, we end up with 3430 images for training and 858 images for validation which is 20% of the original whole training dataset which adds up to 4288 images. 
 
II. 	Model creation and training- 
 
In the second part, we are creating our model and training it using the train and validation data. 
 
1. Model Construction- 
Now we define our base model which is Resnet50v2 and give it our input shape of image and 3 signifying RGB. We initially chose VGG-16, but reverted our decision because of this being more recent and accurate model. This will be our first layer in the model We are developing a sequential mode where layers are added in series one after another. This concept of model creation is called Transfer Learning, in which we use a model trained on millions of images whose classification accuracy is found to be very high for a large number of objects. Then we add a series of dense and dropout layers. Dropout Layers is a simple ay to avoid overfitting in the model. This creates our Convolution Neural Network. 
This model was selected mainly because  
•	ResNets are easy to optimize, but the “plain” networks (that simply stack layers) shows higher training error when the depth increases. 
•	ResNets can easily gain accuracy from greatly increased depth, producing results which are better than previous networks. 
 
 
 III. 
predictions is taken and video is classified by that. We do this for all the test videos. Then we calculate the accuracy which is around 97%. 
 
 
IV. 	Front-end development- 
We did simple front-end development in HTML file with very little CSS.  
•	The submit page is created to upload the video. 
•	Result page to show the prediction. 
 
V. 	Video prediction - 
This part has same code as testing file. The only difference is that we were testing for a lot of videos to measure the accuracy of the model developed, but here we only predict for a single video given by the user. We made it in integration with Flask to deploy it on local host. 
 
 
VI. 	Deploy- 
Local: 
We deploy our model with help of flask on local host. The first function opens the HTML page to upload a video. The submit button calls uploader function which requests the video and stores it on the system. Processing is applied on the caught frames and prediction is made. This prediction is sent to result page and displayed. 
 
Online: 
We explored various platforms for online deployment. Digital Ocean, Heroku, Google and AWS. We explored git and Heroku CLI. Finally, we deployed our model using AWS EC2 instances. Below we mention the approach we took and steps we used and our experience on each of the platforms. 
a) Deployment via Heroku: 
There were many Stages for Deployment via Heroku. 
•	We created PROCFILE to specify open the first file for runtime and its properties. 
•	We created APTFILE to specify the buildpacks and external libraries for download when runtime. 
•	We created requirements.txt file which had all the modules we are using in our project and their version so that they are downloaded in the server while running the code. 
We encountered following major problems/errors: o First there were errors due to various File semantics, version compatibility among many others. We searched web to solve them. o Weights file being a large file of 192 MB cannot be uploaded on github manually. 
We installed git and GIT LFS on our system and uploaded using it. 
o OS error while building the model which was not recognizing the weights file. We had to add buildpack in Heroku and its key which was generated in Github app access token.  
The final problem which forced us to shift to another platform was that the slug size in Heroku is limited to 500mb while we our whole file on server was a compressed 618MB file which included all python files and weights.
 
b) Deployment on Docker and Digital Ocean 
 
Although Docker was creating a virtual environment in my system, using Digital ocean was       infeasible due to its fully paid mode. So even trying it was a bit heavy on pockets. 
 
c) Deployment using AWS 
AWS is very feasible and mostly free platform. We used EC2 instances to deploy our Model. 
Below are the steps we went through for our final deployment. 
•	We created a free instance and chose our AMI to be ubuntu. • We opened the site from all traffic in security groups 
•	We created an SSH key pair for instance. 
•	Then we downloaded 3 software: Putty, Putty Generator and WinSCP 
•	Using Putty generated, we saved a ppk file from pem file which was provided as key. 
This will be used to access our server and network. 
•	Then we uploaded our project files using WinSCP.  
•	Using Putty I logged in Ubuntu and downloaded pi3 on server and the using it, also downloaded the python modules mentioned in requirements.txt file. 
•	But because this was a free instance, its memory on server was limited. So we repeated this for a paid server. This had just a little more memory than the previous one and was light on our pockets also. 
•	After all modules were installed, I turned my server on by running my main app file. 
•	The project is now fully online. 
•	The Public DNS mentioned in AWS instance and host given together gives the link of site. 
•	Every time, I restart my server, the DNS changes. So, the link also changes. 
