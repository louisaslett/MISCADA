FROM ghcr.io/rocker-org/devcontainer/tidyverse:4

RUN install2.r --error --skipinstalled --ncpus -2 \
    DataExplorer \
    GGally \
    data.table \
    doMC \
    dplyr \
    forcats \
    foreach \
    ggforce \
    ggplot2 \
    glmnet \
    glue \
    keras \
    latex2exp \
    markdown \
    mgcv \
    mlr3 \
    mlr3learners \
    mlr3verse \
    modeldata \
    mvtnorm \
    nnet \
    pixmap \
    plotly \
    ranger \
    recipes \
    rnaturalearth \
    rnaturalearthdata \
    rpart \
    rsample \
    skimr \
    tidyverse \
    webshot \
    xgboost \
    yardstick \
    && rm -rf /tmp/downloaded_packages \
    && strip /usr/local/lib/R/site-library/*/libs/*.so

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    libgeos-dev \
    libudunits2-dev \
    libproj-dev \
    libgdal-dev \
    libxt6 \
    libglpk-dev \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install tensorflow \
    && pip install tensorflow-hub tensorflow-datasets scipy requests Pillow h5py pandas pydot \
    && echo "RETICULATE_PYTHON=/usr/bin/python3" >> /usr/local/lib/R/etc/Renviron.site

RUN echo "auth-none=1" >> /etc/rstudio/rserver.conf \
    && echo "USER=rstudio" >> /etc/environment
