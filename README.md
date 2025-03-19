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

## create virtual environment
```
python3 -m venv venv
```

## activate virtual environment:
windows:
``` 
venv\Scripts\activate
```
unix or mac:
```
source venv/bin/activate
```

get out of virtual environment:
```
deactivate
```

## add installed dependencies to file (for installing later)
```
pip freeze > requirements.txt
```

## install any needed dependencies to environment
```
pip install -r requirements.txt
```

run flask:
```
flask --app app.py run
```



