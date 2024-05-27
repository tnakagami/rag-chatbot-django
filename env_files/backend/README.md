# Creation of `.env` file for Django
The sample of `.env` file is shown below:

```bash
DOCKER_APP_ENV=development
DJANGO_SECRET_KEY=abcdefghijklmnopqrstuvwxyz0123456789
DJANGO_WWW_VHOST=www.example.com,sub.example2.com
DJANGO_SUPERUSER_NAME=superuser
DJANGO_SUPERUSER_EMAIL=superuser@backend.local
DJANGO_SUPERUSER_PASSWORD=superuserpassword
```

A function of each environment variable is given below.

| Env   | Function |
| :---- | :---- |
| DOCKER_APP_ENV | Product type, `development` or `production` |
| DJANGO_SECRET_KEY | Django secret key |
| DJANGO_WWW_VHOST | Allowed hosts of Django |
| DJANGO_SUPERUSER_NAME | Username of superuser |
| DJANGO_SUPERUSER_EMAIL | Email of superuser |
| DJANGO_SUPERUSER_PASSWORD | Password of superuser |