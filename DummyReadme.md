#Dummy Environment Tutorial:
The dummy environment is used to test the web interface without GPU support for AI/ML modules.

1. Create the virtual environment (Django):
```
python3.12 -m venv venv
```
Flask (Legacy):
```
python3.12 -m venv venvflask
```
2. Activate the environment:

    - **For CMD**:  
      ```cmd
      venv\Scripts\activate.bat
      ```

    - **For PowerShell**:  
      ```powershell
      venv\Scripts\activate.ps1
      ```

    - **For Bash (Linux/Mac)**:  
      ```bash
      source venv/bin/activate
      ```
Replace venv with venvflask if you want to run flask instead:
- **For Bash (Linux/Mac)**: 
```
source venvflask/bin/activate
```

3. Install the requirement

For Django
```
pip install -r django-packages.txt
```

For Flask
```
pip install -r flask-packages.txt
```

4. Set the openai venv
Create .env and enter the openAI key

5. Run the following:
```
python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```
And then find and copy the folder nltk_data to /venvflask

6. Unzip and Copy clips and images folder 

7. Run database Creation/migrations (Django)
```
python ./easyreadweb/manage.py makemigrations
python ./easyreadweb/manage.py migrate
```

8. Run the server

Django
```
python ./easyreadweb/manage.py runserver 0.0.0.0:5001
```

Flask
```
python run.py
```