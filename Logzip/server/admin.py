from django.contrib import admin

# Register your models here.
from server import models
admin.site.register(models.User)
# admin.site.register(models.UploadFiles)