#!/usr/bin/env python

#Dumbo:

pyspark

business = spark.read.json('Dataset/yelp_academic_dataset_business.json')
review = spark.read.json('Dataset/yelp_academic_dataset_review.json')
user = spark.read.json('Dataset/yelp_academic_dataset_user.json')
tip = spark.read.json('Dataset/yelp_academic_dataset_tip.json')
checkin = spark.read.json('Dataset/yelp_academic_dataset_checkin.json')

business.createOrReplaceTempView('business')
review.createOrReplaceTempView('review')
user.createOrReplaceTempView('user')
tip.createOrReplaceTempView('tip')
checkin.createOrReplaceTempView('checkin')

restaurant_subset = spark.sql('SELECT * FROM business WHERE (categories like "%Restaurants%" or categories like "%Restaurant%" or categories like "%Cafes%" or categories like "%Bakeries%" or categories like "%Bars%" or categories like "%Desserts%" or categories like "%Coffee%" or categories like "%Tea%" or categories like "%Juice Bars%" or categories like "%Smoothies%" or categories like "%Food Trucks%" or categories like "%Caterers%" or categories like "%Coffee Roasteries%" or categories like "%Delicatessen%" or categories like "%Deli%")')
restaurant_subset.createOrReplaceTempView('restaurant_subset')

table1 = spark.sql('SELECT a.*, b.review_id, b.user_id, b.stars as review_stars, b.useful, b.funny, b.cool, b.text, b.date, c.business_checkin_dates from (select * from restaurant_subset) as a left join (select * from review) as b on (a.business_id = b.business_id) left join (select business_id, date as business_checkin_dates from checkin) as c on (a.business_id = c.business_id)')
table1.createOrReplaceTempView('table1')

test= spark.sql('SELECT count(distinct review_id) from table1')
test.createOrReplaceTempView('test')

table2 = spark.sql('SELECT a.*, b.name as user_name, b.review_count as user_review_count, b.yelping_since as user_yelping_since, b.useful as user_useful, b.funny as user_funny, b.cool as user_cool, b.elite as user_elite, b.friends as user_friends, b.fans as user_fans, b.average_stars as user_average_stars, b.compliment_hot as user_compliment_hot, b.compliment_more as user_compliment_more, b.compliment_profile as user_compliment_profile, b.compliment_cute as user_compliment_cute, b.compliment_list as user_compliment_list, b.compliment_note as user_compliment_note, b.compliment_plain as user_compliment_plain, b.compliment_cool as user_compliment_cool, b.compliment_funny as user_compliment_funny, b.compliment_writer as user_compliment_writer, b.compliment_photos as user_compliment_photos from (select * from table1) as a left join (select * from user) as b on (a.user_id = b.user_id)')
table2.createOrReplaceTempView('table2')

table3 = spark.sql('SELECT a.*, b.tip_text, b.tip_date, b.tip_compliment_count from (select * from table2) as a left join (select user_id, business_id, text as tip_text, date as tip_date, compliment_count as tip_compliment_count from tip) as b on (a.business_id = b.business_id and a.user_id = b.user_id)')
table3.createOrReplaceTempView('table3')

restaurant_subset.write.json('hdfs:/user/as12453/pub/business_subset.json')
table3.write.json('hdfs:/user/as12453/pub/all_data_restauants.json')

user_count = spark.sql('select user_id from (SELECT user_id, count(distinct review_id) as x from table3 group by user_id) where x >= 15')
user_count.createOrReplaceTempView('user_count')

random_users = spark.sql('select user_id from user_count ORDER BY RAND() limit 10000')
random_users.createOrReplaceTempView('random_users')

table4 = spark.sql('SELECT * from table3 where user_id in (select distinct user_id from random_users )')
table4.createOrReplaceTempView('table4')
#400K reviews

table4.write.json('hdfs:/user/as12453/pub/all_data_restauants_subset.json')