FROM ghcr.io/rocker-org/devcontainer/tidyverse:4
RUN install2.r --error --skipinstalled --ncpus -2 \
    GGally \
    data.table \
    doMC \
    dplyr \
    forcats \
    foreach \
    ggforce \
    ggplot2 \
    glue \
    keras \
    latex2exp \
    mgcv \
    mlr3 \
    mlr3learners \
    mlr3verse \
    mvtnorm \
    nnet \
    pixmap \
    plotly \
    ranger \
    recipes \
    rgeos \
    rnaturalearth \
    rnaturalearthdata \
    rpart \
    rsample \
    skimr \
    tidyverse \
    webshot \
    && rm -rf /tmp/downloaded_packages \
    && strip /usr/local/lib/R/site-library/*/libs/*.so
