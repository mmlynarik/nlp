"""NLP project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))

To add a new path, first import the app:
import blog

Then add the new path:
path('blog/', blog.urls, name="blog")

Learn more here: https://docs.djangoproject.com/en/2.1/topics/http/urls/
"""

from django.contrib import admin
from django.urls import path, include

admin.autodiscover()

import hello.views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", hello.views.index, name="index"),
    path("db/", hello.views.db, name="db"),
]
