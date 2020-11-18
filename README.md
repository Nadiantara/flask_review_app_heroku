# echo-flask-web
Migrated Echo flask web repo. Previously on echo-project-internal

## **Running the project on local:**
### **Before you run the commands**
  - Make sure you do the commands on the project directory.
  - Make sure you have the env files (contact Ardy or Nadi from Unit 2)
  
There are two options to run the project:
#### **1. With Virtual Environment**
- Create a virtual environment:
  - `python -m venv venv`
- activate the virtual environment
  - `.venv/Scripts/activate`
- Install Dependencies: 
  - With PIP: `pip install -r requirements.txt`
  - ~~- With Anaconda: `conda install --file requirements.txt`~~ Recommended using PIP
- Run the web app with
  - `flask run`

#### **2. With Docker**
- Build docker images
  -  `docker-compose build` (This will take some time)
- Run the services
  -  `docker-compose up`

## **Deploying the project:**

### **Option #1: Deployment with DB Server**
Big thanks to:
  - Adam Mahendra for the mentorship
  - [Samuel Chan](https://github.com/onlyphantom/) for the original guide

The project is deployed on Azure. To deploy the project follow these steps:

**Note**: 
- Those that are enclosed with tag {} are just "variables". Change to the names you see fit.
- We try to make every "variable" name unique. So if you see a "variable" with the same name in different commands, it's the same basically the same thing across commands

<br>

### Step 1: Creating a Deployment User

In my Azure Portal, click on the cloud shell icon to bring up cloud shell. The following commands are executed in the Azure CLI (bash option)

<pre class="code CodeMirror" data-line="5">          
az webapp deployment user set --user-name {yourUsername} --password {yourPassword} 
</pre>

<br>

### Step 2: Create Resource Group and Azure Database for Postgres
We use postgreSQL for database

<pre class="code CodeMirror" data-line="11">          
`az group create --name {yourResourceGroup} --location "Southeast Asia"` 
</pre>

Note: Southeast Asia is chosen because we're in Indonesia. Pick the location closest to your users

<pre class="code CodeMirror" data-line="17">          
`az postgres server create --resource-group {yourResourceGroup} --name {your} --location "Southeast Asia" --admin-user {yourDBAdminUserName} --admin-password {yourDBAdminUserPassword}` 
</pre>

Create firewall rules for the postgres server. This grants access to the database from Azure resources:

<pre class="code CodeMirror" data-line="22">          
  az postgres server firewall-rule create --name allAzureIPs --server {yourDBName} --resource-group {yourResourceGroup} --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0` 
        </pre>

<br>

### Step 3: Create an Azure App Service plan

The following example creates an App Service plan named`yourAppServicePlan`in the**Basic**pricing tier (`--sku B1`) and in a Linux container (`--is-linux`).

Note:
- [Click here](https://azure.microsoft.com/en-gb/pricing/details/app-service/linux/) to see Azure's pricing details 
- My opinions on choosing your tier:
  - Free is **laggy and limited**. You can use it to test whether deployment is successful or not, but **very not** recommended to use to test your app's general performance
  - Basic can run well to test your app's general performance. But not so ready to go production
  - Above that should run fine, but costly

<pre class="code CodeMirror" data-line="30">          
`az appservice plan create --name {yourAppServicePlan} --resource-group {yourResourceGroup} --sku B1 --is-linux` 
        </pre>

If request succeeded, after 15 seconds it will return:

<pre class="code CodeMirror" data-line="35">
{
  "adminSiteName": null,
  "freeOfferExpirationTime": "2018-12-26T12:03:27.050000",
  "geoRegion": "Southeast Asia",
  "hostingEnvironmentProfile": null,
  "hyperV": false,
  < JSON data removed for brevity>,
} 
</pre>

<br>

### Step 4: Create a Web App

Create a web app in the `yourAppServicePlan` App Service plan.

In the following example, replace`<app_name>`with a globally unique app name (valid characters are`a-z`,`0-9`, and`-`). For our app, the runtime is set to`PYTHON|3.7`.

<pre class="code CodeMirror" data-line="56">          
`az webapp create --resource-group {yourResourceGroup} --plan {yourAppServicePlan} --name {yourAppName} --runtime "PYTHON|3.7" --deployment-local-git` 
        </pre>

It returns:

<pre class="code CodeMirror" data-line="61">
`Local git is configured with url of {your_azure_git_link}
{
  "availabilityState": "Normal",
  "clientAffinityEnabled": true,
  "clientCertEnabled": false,
 ...
  "name": "Pedagogy",
  "resourceGroup": "xxxxxxxx"
}` 
        </pre>

We’ve created an empty new web app, with git deployment enabled.

<br>

### Step 5: Configure environment variables

Our app requires some environment variables to work, and we can issue the following command to create them:

<pre class="code CodeMirror" data-line="78">          
az webapp config appsettings set --name {yourAppName} --resource-group {yourResourceGroup} --settings {copy paste the contents of db.env here}
</pre>

The URL of the Git remote is shown in the`deploymentLocalGitUrl`property, with the format`https://<username>@<app_name>.scm.azurewebsites.net/<app_name>.git`. Save this URL as we need it later.

Navigate to `<appname>.azurewebsites.net` to see the initial page of your app.

<br>

### Step 6: Push to Azure from Git

Back in the local terminal, we want to add an Azure remote to our local Git repository, then push to the Azure remote to deploy our app. We also need a certain **application.py** for the deployment engine so let’s add that too:

<pre class="code CodeMirror" data-line="95"> 
`touch application.py
git add .
git commit -m 'initial push'
git remote add azure {your_azure_git_link}
git push azure master` 
</pre>

During the push stage, when prompted for credentials by Git Credential Manager, make sure that you enter the credentials you created in the creation of Deployment User phase (Step 1), not the credentials we use to sign in to the Azure portal.

Anytime we add code and want to push the changes to Azure, we issue the familiar commit-push command sequence:

<pre class="code CodeMirror" data-line="107"> 
git commit -am "updated output"
git push azure master 
</pre>

<br>

### Step 7: Configure entry point

By default, the built-in image looks for a _wsgi.py_ or _application.py_ in the root directory as the entry point, but our entry point is flask_test/**init.py**

The_application.py_ we added earlier is empty and does nothing. Use `az webapp config set` to set a statup script:

<pre class="code CodeMirror" data-line="118">          
webapp config set --name {yourAppName} --resource-group {yourResourceGroup} --startup-file "gunicorn --bind :8000 flask_test:app"
</pre>

The Gunicorn format follows this convention:

<pre class="code CodeMirror" data-line="123">          
`gunicorn --bind :8000 --chdir /home/site/wwwroot/<subdirectory> <module>:<variable>` 
</pre>

`--chdir` is required if our entry point is not in the root directory and`<subdirectory>`is the subdirectory.`<module>`is the name of the _.py_ file and`<variable>`is the variable in the module that represents your web app.

<br>

### Step 8: Update model

Any time we make code changes, we can deploy to azure by following the simple commit-push sequence:

<pre class="code CodeMirror" data-line="132">
git add . 
git commit -m "updated data model" 
git push azure master
</pre>

or:

<pre class="code CodeMirror" data="138">         
git -am 
</pre>

### **Option #2: Deployment with Docker**

**Still TBD**
We've deployed it on Azure but it's not yet working haha

**Why Docker?**
- Very simple local setup, as long as you have docker installed in your machine
- Integrate various tech stacks easily. We plan to use these with docker:
  - NGINX: Make app more performant
  - Postgres
  - possibly other stacks
