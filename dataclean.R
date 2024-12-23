library(dplyr)     
library(readr)    
library(lubridate)
library(zoo)

train = read.csv("train.csv")
test = read.csv("test.csv")
stores = read.csv("stores.csv")
oil = read.csv("oil.csv")
holidays_events = read.csv("holidays_events.csv")

train = rename(train, "product_type" = "family")
train = rename(train, "store_id" = "store_nbr")
stores = rename(stores, "store_id" = "store_nbr")
oil = rename(oil, "oil_price" = "dcoilwtico")
test = rename(test, "product_type" = "family")
test = rename(test, "store_id" = "store_nbr")

train = left_join(train, stores, by = "store_id")
train = left_join(train, oil, by = "date")
train = left_join(train, holidays_events, by = "date")
test = left_join(test, stores, by = "store_id")
test = left_join(test, oil, by = "date")
test = left_join(test, holidays_events, by = "date")

train$year = year(train$date)
train$month = month(train$date)
train$day_of_week = wday(train$date, label = TRUE)
test$year = year(test$date)
test$month = month(test$date)
test$day_of_week = wday(test$date, label = TRUE)

train = train[order(train$store_id, train$product_type, train$date), ]
train$sales_lag_1 = ave(train$sales, train$store_id, train$product_type, FUN = function(x) lag(x, 1))
train$sales_lag_7 = ave(train$sales, train$store_id, train$product_type, FUN = function(x) lag(x, 7))

train$rolling_avg_7 = ave(train$sales, train$store_id, train$product_type, FUN = function(x) rollmean(x, k = 7, fill = NA, align = "right"))

write_csv(train, "combined_train.csv")
write_csv(test, "test.csv")
