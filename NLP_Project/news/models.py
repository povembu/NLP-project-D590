from django.db import models

# Create your models here.
class News(models.Model):
    comment = models.CharField(("comment"), max_length=255)
    label = models.IntegerField("label") # contain only 0 or 1
    datetime = models.DateTimeField("datetime")