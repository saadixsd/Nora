## git commands
### get project files into your computer
```
git clone https://github.com/saadixsd/Nora.git
```
### create a new branch and go into it
```
git branch [branch-name]
git checkout [branch-name]
```

### put code into a branch
```
git add .
git commit -m "commit-message"
git pull origin [branch-name]
git push origin [branch-name]
```

## virtual environment commands
### create virtual environment
```
python3 -m venv venv
```

### activate virtual environment:
windows:
``` 
venv\Scripts\activate
```
unix or mac:
```
source venv/bin/activate
```

### get out of virtual environment:
```
deactivate
```

### add installed dependencies to file (for installing later)
```
pip freeze > requirements.txt
```

### install any needed dependencies to environment
```
pip install -r requirements.txt
```

## run app commands
### run flask:
```
flask --app app.py run
```

### run react:
```
npm run start
```



