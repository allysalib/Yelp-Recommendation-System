#!/usr/bin/env python

pyspark

business = spark.read.json('scratch/as12453/Dataset/yelp_academic_dataset_business.json')
review = spark.read.json('scratch/as12453/yelp_academic_dataset_review.json')
user = spark.read.json('scratch/as12453/Dataset/yelp_academic_dataset_user.json')
tip = spark.read.json('scratch/as12453/Dataset/yelp_academic_dataset_tip.json')
checkin = spark.read.json('scratch/as12453/Dataset/yelp_academic_dataset_checkin.json')

business.createOrReplaceTempView('business')
review.createOrReplaceTempView('review')
user.createOrReplaceTempView('user')
tip.createOrReplaceTempView('tip')
checkin.createOrReplaceTempView('checkin')

restaurant_subset = spark.sql('SELECT * FROM business WHERE (category like '%Restaurants%' or category like '%Restaurant%' or category like '%Cafes%' or category like '%Bakeries%' or category like '%Bars%' or category like '%Desserts%' or category like '%Coffee & Tea%' or category like '%Juice Bars & Smoothies%' or category like '%Food Trucks%' or category like '%Caterers%' or category like '%Coffee Roasteries%' or category like '%Delicatessen%' or category like '%Deli%')')
restaurant_subset.createOrReplaceTempView('restaurant_subset')

table1 = spark.sql('SELECT a.*, b.review_id, b.user_id, b.stars, b.useful, b.funny, b.cool, b.text, b.date, c.business_checkin_dates from (select * from restaurant_subset) as a left join (select * from review) as b on (a.business_id = b.business_id) left join (select business_id, date as business_checkin_dates from checkin) as c on (a.business_id = c.business_id)')
table1.createOrReplaceTempView('table1')

table2 = spark.sql('SELECT a.*, b.user_name, b.user_review_count, b.user_yelping_since, b.user_useful, b.user_funny, b.user_cool, b.user_elite, b.user_friends, b.user_fans, b.user_average_stars, b.user_compliment_hot, b.user_compliment_more, b.user_compliment_profile, b.user_compliment_cute, b.user_compliment_list, b.user_compliment_note, b.user_compliment_plain, b.user_compliment_cool, b.user_compliment_funny, b.user_compliment_writer, b.user_compliment_photos from (select * from table1) as a left join (select user_id, user_name, user_review_count, user_yelping_since, user_useful, user_funny, user_cool, user_elite, user_friends, user_fans, user_average_stars, user_compliment_hot, user_compliment_more, user_compliment_profile, user_compliment_cute, user_compliment_list, user_compliment_note, user_compliment_plain, user_compliment_cool, user_compliment_funny, user_compliment_writer, user_compliment_photos from user) as b on (a.user_id = b.user_id)')
table2.createOrReplaceTempView('table2')

table3 = spark.sql('SELECT a.*, b.tip_text, b.tip_date, b.tip_compliment_count from (select * from table2) as a left join (select user_id, business_id, text as tip_text, date as tip_date, compliment_count as tip_compliment_count from tip) as b on (a.business_id = b.business_id and a.user_id = b.user_id')
table3.createOrReplaceTempView('table3')

table3.write.csv('scratch/as12453/Dataset/full_restauant_subset.csv')
