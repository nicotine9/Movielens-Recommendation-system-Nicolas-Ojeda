packages <- c("tidyverse", "caret", "data.table", "dslabs", "stringr", "forcats", "ggplot2", "lubridate")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/movies.dat"))),
                col.names = c("movieId", "title", "genres"))

set.seed(1)  # For reproducibility
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]



# Separate genres into individual rows for analysis
movielens_separated_genres <- movielens %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(average_rating = mean(rating, na.rm = TRUE)) %>%
  ungroup() %>%
  arrange(desc(average_rating))

# Plot the average rating by genre
ggplot(movielens_separated_genres, aes(x = reorder(genres, average_rating), y = average_rating)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +  # Flip coordinates to make the genres list vertically
  theme(axis.text.y = element_text(angle = 0, hjust = 1)) +  # Ensure genre labels are horizontal
  labs(x = "Genre", y = "Average Rating", title = "Average Movie Rating by Genre")

# Distribution of movie ratings
ggplot(movielens, aes(x = rating)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  ggtitle("Distribution of Movie Ratings")




# Number of ratings per user
user_ratings <- movielens %>% 
  group_by(userId) %>%
  summarize(n_ratings = n()) %>%
  ungroup()
# Visualize the distribution of number of ratings per user
ggplot(user_ratings, aes(x = n_ratings)) +
  geom_histogram(bins = 50, fill = "cornflowerblue") +
  ggtitle("Number of Ratings per User")
# Number of ratings per movie
movie_ratings_count <- movielens %>% 
  group_by(title) %>%
  summarize(n_ratings = n()) %>%
  arrange(desc(n_ratings))

# Top 10 most rated movies
top_movies <- head(movie_ratings_count, 10)
ggplot(top_movies, aes(x = reorder(title, n_ratings), y = n_ratings)) +
  geom_bar(stat = "identity", fill = "tomato") +
  coord_flip() +
  labs(x = "Movie Title", y = "Number of Ratings") +
  ggtitle("Top 10 Most Rated Movies")
# Convert the timestamp to a readable date format
movielens$date <- as.Date(as.POSIXct(movielens$timestamp, origin = "1970-01-01"))


# Average rating over time
ratings_over_time <- movielens %>% 
  group_by(date) %>%
  summarize(average_rating = mean(rating))

ggplot(ratings_over_time, aes(x = date, y = average_rating)) +
  geom_line(color = "steelblue") +
  ggtitle("Average Movie Rating Over Time")
# Splitting genres into separate rows for analysis
movielens <- movielens %>% 
  mutate(genres = strsplit(as.character(genres), "\\|")) %>% 
  unnest(genres)


# Calculate the global average rating
global_average <- mean(train_set$rating)

# Naive model prediction is just the global average for all ratings
naive_predictions <- rep(global_average, nrow(test_set))

# Calculate RMSE for Naive Model
naive_rmse <- sqrt(mean((test_set$rating - naive_predictions)^2))




# Calculate the average rating deviation for each movie
movie_effects <- train_set %>% 
  group_by(movieId) %>%
  summarize(movie_effect = mean(rating) - global_average) %>%
  ungroup() 

# Predictions are the global average plus the movie effect
movie_effect_predictions <- test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  mutate(prediction = global_average + movie_effect) %>%
  .$prediction

# Calculate RMSE for Movie Effect Model
movie_effect_rmse <- sqrt(mean((test_set$rating - movie_effect_predictions)^2, na.rm = TRUE))



# Calculate the user effect
user_effects <- train_set %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(user_effect = mean(rating - movie_effect - global_average)) %>%
  ungroup()

# Predictions are the global average plus movie and user effects
user_effect_predictions <- test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(prediction = global_average + movie_effect + user_effect) %>%
  .$prediction

# Calculate RMSE for User Effects Model
user_effect_rmse <- sqrt(mean((test_set$rating - user_effect_predictions)^2, na.rm = TRUE))



# Split genres for analysis
movielens_genres <- movielens %>% 
  separate_rows(genres, sep = "\\|") 

# Average rating per genre
average_genre_ratings <- movielens_genres %>%
  group_by(genres) %>%
  summarize(average_rating = mean(rating), total_ratings = n()) %>%
  arrange(desc(average_rating))

# Visualizing average rating per genre
ggplot(average_genre_ratings, aes(x = reorder(genres, average_rating), y = average_rating)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Average Movie Rating by Genre", x = "Genre", y = "Average Rating")


# Converting timestamp to a readable date format
movielens$date <- as.Date(as.POSIXct(movielens$timestamp, origin = "1970-01-01"))

# Analyzing rating trends over time
rating_trends <- movielens %>%
  group_by(date) %>%
  summarize(average_rating = mean(rating))

ggplot(rating_trends, aes(x = date, y = average_rating)) +
  geom_line(color = "darkgreen") +
  labs(title = "Temporal Trends in Movie Ratings", x = "Date", y = "Average Rating")


# Extract release year from the movie title
movielens$release_year <- as.numeric(str_extract(movielens$title, "\\d{4}$"))

# Analyzing the impact of release year on ratings
yearly_rating_trends <- movielens %>%
  group_by(release_year) %>%
  summarize(average_rating = mean(rating))

ggplot(yearly_rating_trends, aes(x = release_year, y = average_rating)) +
  geom_line(color = "blue") +
  labs(title = "Impact of Release Year on Movie Ratings", x = "Release Year", y = "Average Rating")



# Define RMSE calculation function
calculate_rmse <- function(actual, predicted) {
  # Ensure that actual and predicted are numeric vectors of the same length
  if (length(actual) != length(predicted)) {
    stop("The lengths of actual and predicted ratings must be equal.")
  }
  
  # Calculate the RMSE
  rmse <- sqrt(mean((actual - predicted) ^ 2, na.rm = TRUE))
  return(rmse)
}

# Usage of RMSE calculation function
# Assuming 'test_set$rating' contains the actual ratings
# And 'naive_predictions', 'movie_effect_predictions', 'user_effect_predictions' contain the model predictions

# Calculate RMSE for the Naive Model
naive_rmse <- calculate_rmse(test_set$rating, naive_predictions)

# Calculate RMSE for the Movie Effect Model
movie_effect_rmse <- calculate_rmse(test_set$rating, movie_effect_predictions)

# Calculate RMSE for the User Effects Model
user_effect_rmse <- calculate_rmse(test_set$rating, user_effect_predictions)

# Output RMSE for each model
cat("RMSE for Naive Model: ", naive_rmse, "\n")
cat("RMSE for Movie Effect Model: ", movie_effect_rmse, "\n")
cat("RMSE for User Effects Model: ", user_effect_rmse, "\n")


# Distribution of movie ratings
ggplot(movielens, aes(x = rating, fill = factor(rating))) +
  geom_histogram(bins = 30, color = "black") +
  labs(title = "Distribution of Movie Ratings", x = "Rating") +
  scale_fill_brewer(palette = "Set1")  # Different colors for each rating




# Number of ratings per user
user_ratings <- movielens %>%
  group_by(userId) %>%
  summarize(n_ratings = n(), .groups = 'drop')

# Visualize the distribution of number of ratings per user
user_ratings_hist <- ggplot(user_ratings, aes(x = n_ratings)) +
  geom_histogram(bins = 50, fill = "cornflowerblue") +
  labs(title = "Number of Ratings per User", x = "Number of Ratings", y = "Count")

# Number of ratings per movie
movie_ratings_count <- movielens %>%
  group_by(title) %>%
  summarize(n_ratings = n(), .groups = 'drop') %>%
  arrange(desc(n_ratings))

# Select the top 20 most-rated movies to prevent overcrowding in the plot
top_movies <- head(movie_ratings_count, 20)

# Visualize the top 20 most-rated movies
movie_ratings_bar <- ggplot(top_movies, aes(x = reorder(title, n_ratings), y = n_ratings)) +
  geom_bar(stat = "identity", fill = "tomato") +
  coord_flip() +  # Flip the coordinates to make the movie titles readable
  labs(title = "Top 20 Most Rated Movies", x = "Movie Title", y = "Number of Ratings") +
  theme(axis.text.y = element_text(size=8))  # Adjust text size for readability

# Print the plots
print(user_ratings_hist)
print(movie_ratings_bar)



# Convert timestamp to date
movielens$date <- as.Date(as.POSIXct(movielens$timestamp, origin = "1970-01-01"))

# Average rating over time
ratings_over_time <- movielens %>% 
  group_by(date) %>%
  summarize(average_rating = mean(rating))

# Line chart for temporal trends
ggplot(ratings_over_time, aes(x = date, y = average_rating)) +
  geom_line(color = "steelblue") +
  labs(title = "Average Movie Rating Over Time", x = "Date", y = "Average Rating")




# Average rating for each genre
genre_ratings <- movielens %>%
  group_by(genres) %>%
  summarize(average_rating = mean(rating), n = n()) %>%
  arrange(desc(average_rating))

# Bar chart for average rating by genre
ggplot(genre_ratings, aes(x = reorder(genres, average_rating), y = average_rating)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  coord_flip() +
  labs(title = "Average Rating by Genre", x = "Genre", y = "Average Rating")


# RMSE values for different models
rmse_values <- data.frame(
  Model = c("Naive Model", "Movie Effect Model", "User Effects Model"),
  RMSE = c(naive_rmse, movie_effect_rmse, user_effect_rmse)
)

# Create a side-by-side bar chart
ggplot(rmse_values, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model RMSE Comparison", x = "Model", y = "RMSE") +
  theme_minimal() +
  theme(legend.position = "none")