# Backend
## Preparations
### Step1: Run makemigrations and migrate
Migrations are how Django stores changes to your models.
To do this, from the command line, run the following command, where "app-name" is a Django's application name.

```bash
# In the host environment
docker-compose run --rm -it --entrypoint /bin/bash backend
# In the container environment
python manage.py makemigrations app-name
```

By running makemigrations, you're telling Django that you've made some changes to your models and that you'd like the chages to be stored as a migration.

There's a command that will run the migrations for you and manage your database schema automatically - that's called migrate. Now, run migrate to create your model tables in your database.

```bash
python manage.py migrate
```

Please remember the tree-step guid to making model changes:

1. Change your models (in models.py).
1. Run `python manage.py makemigrations app-name` to create migrations for those changes in your application.
1. Run `python manage.py migrate` to apply those changes to the database.

### Step2: Create superuser
To create superuser account, let's run following command, where `DJANGO_SUPERUSER_NAME`, `DJANGO_SUPERUSER_EMAIL`, and `DJANGO_SUPERUSER_PASSWORD` are environment variables defined by `env_files/backend/.env`.

```bash
python manage.py custom_createsuperuser --username ${DJANGO_SUPERUSER_NAME} --email ${DJANGO_SUPERUSER_EMAIL} --password ${DJANGO_SUPERUSER_PASSWORD}
```

Finally, execute the following command to exit the docker environment.

```bash
exit # or press Ctrl + D
```

### Step3: Customize Django application
In the [backend/app](./app) directory, store Python scripts. Please modify those scripts if you want to customize them.

### Step4: Create multilingual localization messsages
Run the following commands to reflect translation messages.

```bash
# In the host environment
docker-compose run --rm -it --entrypoint django-admin makemessages -l ja backend
#
# Edit .po files by using your favorite editor (e.g. vim)
#
docker-compose run --rm -it --entrypoint django-admin compilemessages backend
```

## Test
### Preparations
In this project, `pytest` and pytest's third packages are used. In particular, `pytest-django` is useful when I develop web applications by using the Django framework.

So I prepare `conftest.py` in the top directory of `all_tests`. The details are as follows:

```python
# all_tests/conftest.py
import pytest

@pytest.fixture(scope='session', autouse=True)
def django_db_setup(django_db_setup):
  pass
```

Then, I create test scripts for each application. See [this directory](./app/all_tests) in detail.

### Execution
Enter the following command on your terminal, then execute `pytest` command.

```bash
# In the host environment
docker-compose run --rm -it --entrypoint /bin/bash backend
# In the docker environment
pytest
```