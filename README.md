
# Hi, I'm Ashutosh ! 👋
In this project, we are utilizing ML models to forecast Airbnb property pricing and identify the key attributes that will have the biggest effects on the house.
Additionally, we are attempting to understand the model by using explainable AI LIME and SHAP. There are two other sections in the code in addition to the three machine learning models:

1.) Prediction explanation with LIME
2.) Prediction explanation with SHAMP




# AirBnb Price Prediction- ML and Explainable AI (LIME and SHAP)

We use the Inside Airbnb dataset to identify significant trends in customer interest and predict the pricing of rentals in the US and Europe. This problem is similar to a classical use case of machine learning: house price prediction. We can then provide solutions to business difficulties.
There are several questions that we want to find answers to:

●	How to predict the price for each listing?

●	Can we find out how the features of listings affect the price locally and globally?

## 🔗 Links
[Code link](https://github.com/Abhiashu10/Airbnb-Price-Prediction--LIME-and-SHAP/blob/d024360cda85a95a689e8876cc4756bfac2e4c45/Airbnb%20Price%20Prediction-Explainable%20AI.ipynb)

## Features
There are 27 total Input features in the dataset: -

➢	accommodates: The maximum capacity of the listing.   
➢	amenities: Furniture that the listing has.   
➢	availability_365: The availability of the listing 365 days in the future as determined by the calendar. Note a listing may not be available because it has been booked by a guest or blocked by the host.   
➢	bedrooms: The number of bedrooms.  
➢	beds: The number of bed(s).  
➢	calculated_host_listings_count: The number of listings the host has in the current scrape, in the city/region geography.  
➢	host_has_profile_pic: boolean [t=true; f=false].  
➢	host_id: Airbnb's unique identifier for the host/user.  
➢	host_identity_verified: boolean [t=true; f=false].   
➢	host_name: Name of the host. Usually just the first name(s).   
➢	host_response_rate: the rate the host respond to their messages.   
➢	host_response_time: the amount of time the host takes to reply to their messages.   
➢	id: Airbnb's unique identifier for the listing.    
➢	instant_bookable: [t=true; f=false]. Whether the guest can automatically    book the listing without the host requiring to accept their booking request.   
➢	last_review: The date of the last/newest review.   
➢	latitude: Uses the World Geodetic System (WGS84) projection for latitude and longitude.    
➢	longitude: Uses the World Geodetic System (WGS84) projection for latitude and longitude.     
➢	maximum_nights: maximum number of night stay for the listing (calendar rules may be different).    
➢	minimum_nights: minimum number of night stay for the listing (calendar rules may be different).    
➢	name: Name of the listing.     
➢	neighborhood: the neighborhood of the listing.    
➢	number_of_reviews: The number of reviews the listing has.    
➢	property_type: Self-selected property type. Hotels and Bed and Breakfasts are described as such by their hosts in this field.    
➢	reviews_per_month: The number of reviews the listing has over the lifetime of the listing.    
➢	review_scores_rating: average rating.    
➢	room_type: [ Entire home/apt | Private room | Shared room | Hotel ]

# Target variable:
price: the daily price of a listing. 


# Categorical features
We get categorical features, evaluate them and convert them to numerical features. Those categorical features are amenities, host_has_profile_pic, host_identity_verified, host_name, host_response_rate, host_response_time, instant_bookable, last_review, name, neighborhood, price, property_type and room_type.
amenities: because they are lists of amenities the listings have, we try to convert them to the number of amenities those listings have

host_has_profile_pic, host_identity_verified, instant_bookable: because these features have 2 boolean values ‘t’, ‘f’ so we convert them into binary values ‘0’, ‘1’.

# Numerical features
We get all numeric features, including categorical features that we turn into numeric features. Then we deal with outliers. At first, we try to use InterQuartile Range to detect and remove outliers of each feature that are outside of Q1 - 1.5 * IQR  and Q3 + 1.5 * IQR. However, this step removes so many outliers that the dataset reduces to only 50,000 observations left and decreases the accuracies of models we build later. So we decided to modify the range a little bit. We calculate the upper bound (97.5%) and lower bound (2.5 %) of values of each feature, and remove observations that have values outside those bounds. After doing this step, the remaining dataset has around 150,000 observations, which is pretty suitable to train and test ML models.









