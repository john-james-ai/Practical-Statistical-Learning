library(lubridate)
library(tidyverse)
library(magrittr)

inspect = function(x) {
  dim(x)
  names(x)
  x %>% head() %>% kable()  
}
