import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors

# Hard-coded K-Means class
class HardCodedKMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, data):
        np.random.seed(42)
        initial_indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        self.centroids = data.iloc[initial_indices].values

        for _ in range(100):
            self.labels = np.array([self._closest_centroid(row) for row in data.values])
            new_centroids = np.array([data[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def _closest_centroid(self, row):
        distances = np.linalg.norm(self.centroids - row, axis=1)
        return np.argmin(distances)

class HardCodedSentimentAnalyzer:
    def __init__(self):
        # Sample word lists for 'positive' and 'negative' sentiments
        self.positive_words = ['love', 'good', 'excellent', 'happy', 'joy', 'awesome', 'great', 'friend', 'like', 'enjoy']
        self.negative_words = ['bad', 'sad', 'terrible', 'hate', 'angry', 'awful', 'poor', 'dislike', 'pain', 'bored']

        # Calculate word counts for each sentiment category
        self.dict_positive = self._count_words(self.positive_words)
        self.dict_negative = self._count_words(self.negative_words)

        # Combine all unique words for likelihood calculation
        self.unique_words = set(self.positive_words + self.negative_words)

        # Prior probabilities (these can be adjusted based on dataset)
        self.prob_positive = 0.6
        self.prob_negative = 0.4

        # Create the likelihood table
        self.df_likelihood = self._create_likelihood_table()

        # Initialize to store the latest user input
        self.user_input = None

    def _count_words(self, words):
        word_count = {}
        for word in words:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
        return word_count

    def _create_likelihood_table(self):
        likelihood_data = []
        for word in self.unique_words:
            positive_prob = self.dict_positive.get(word, 0) / len(self.positive_words)
            negative_prob = self.dict_negative.get(word, 0) / len(self.negative_words)
            likelihood_data.append([word, positive_prob, negative_prob])

        # Create a DataFrame
        return pd.DataFrame(likelihood_data, columns=['Word', 'P(word|positive)', 'P(word|negative)'])

    def set_user_input(self, text):
        """Store the latest user input for sentiment analysis."""
        self.user_input = text

    def predict_sentiment(self):
        """Analyze the stored user input for sentiment prediction."""
        if self.user_input is None:
            return "No input provided."

        # Calculate initial scores based on prior probabilities
        positive_score = self.prob_positive
        negative_score = self.prob_negative

        # Split stored user input text
        user_words = self.user_input.split(" ")

        word_found = False
        for keyword in user_words:
            for _, row in self.df_likelihood.iterrows():
                if keyword == row['Word']:
                    word_found = True
                    positive_score *= row['P(word|positive)']
                    negative_score *= row['P(word|negative)']
                    break

        if not word_found:
            return "Neutral or Unrecognized sentiment"

        # Determine sentiment based on the final scores
        if positive_score > negative_score:
            return "Positive sentiment"
        else:
            return "Negative sentiment"



class HardCodedLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Update weights and bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_classified = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_classified)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))






# Decision Tree class
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree_ = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree_.fit(X, y)

    def predict(self, X):
        return self.tree_.predict(X)

    def predict_proba(self, X):
        return self.tree_.predict_proba(X)

    def plot(self):
        plt.figure(figsize=(12, 8))
        plot_tree(self.tree_, filled=True)
        plt.show()



st.set_page_config(page_title="Book Recommender System", layout="wide")

st.title('📚 Book Recommender System')
st.subheader('Discover your next favorite book!')

# Load datasets
books = pd.read_csv('https://drive.google.com/uc?id=1dK8nN1nhhf9KrVnWA3gyd-uGpzlafZWo')
ratings = pd.read_csv('https://drive.google.com/uc?id=1rLqmW7ZxMzJlS8e09yiyeZltc12lkFOo')

# Limit ratings for performance
ratings = ratings.head(10000)

# Prepare user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

# Create book name list for selection
book_name_list = books['title'].tolist()

# K-Means Clustering for users
kmeans = HardCodedKMeans(n_clusters=5)
kmeans.fit(user_item_matrix)

analyzer = HardCodedSentimentAnalyzer()

# Prepare classification data
def prepare_classification_data(ratings, books):
    merged = pd.merge(ratings, books[['book_id', 'average_rating']], on='book_id')
    merged['liked'] = np.where(merged['rating'] >= 4, 1, 0)
    features = merged[['user_id', 'book_id', 'average_rating']]
    labels = merged['liked']
    return features, labels

# Train classifiers
features, labels = prepare_classification_data(ratings, books)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize classifiers
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)

clf_svm = SVC(probability=True)
clf_svm.fit(X_train, y_train)

clf_hc_dt = DecisionTree(max_depth=5)
clf_hc_dt.fit(X_train, y_train)


clf_hc_lr = HardCodedLogisticRegression()
clf_hc_lr.fit(X_train[['user_id', 'book_id', 'average_rating']].values, y_train.values)


def get_top_rated_books(books, ratings, n=5):
    top_books = ratings.groupby('book_id').agg({'rating': 'mean'}).reset_index()
    top_books = top_books.merge(books[['book_id', 'title', 'image_url']], on='book_id')
    top_books = top_books.sort_values(by='rating', ascending=False).head(n)
    return top_books[['title', 'image_url']]

# Function to calculate ROC curves
def calculate_roc_curves(X_test, y_test, clf_dict):
    roc_curves = {}
    for name, clf in clf_dict.items():
        y_scores = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_curves[name] = (fpr, tpr, roc_auc)
    return roc_curves

# Function to plot ROC curves
def plot_roc_curves(roc_curves):
    plt.figure(figsize=(10, 6))
    for name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Function to predict if a user will like a book
def predict_user_like(user_id, book_id, model):
    if user_id not in features['user_id'].values or book_id not in features['book_id'].values:
        return None

    average_rating = books.loc[books['book_id'] == book_id, 'average_rating'].values[0]
    input_features = np.array([[user_id, book_id, average_rating]])

    prediction = model.predict(input_features)
    return prediction[0]

# Function to get K-Means recommendations for a specific user
def get_recommendations_for_user(user_id):
    user_index = user_item_matrix.index.get_loc(user_id)
    user_cluster = kmeans.labels[user_index]
    cluster_users = user_item_matrix.index[kmeans.labels == user_cluster]

    cluster_books = ratings[ratings['user_id'].isin(cluster_users)]
    top_books = cluster_books.groupby('book_id').size().reset_index(name='counts')
    top_books = top_books.nlargest(5, 'counts')

    recommended_books = books[books['book_id'].isin(top_books['book_id'])][['title', 'image_url']]

    return {
        'recommended_books': recommended_books['title'].tolist(),
        'image_urls': recommended_books['image_url'].tolist()
    }

# KNN Recommendations with distance plot
def get_knn_recommendations(selected_book):
    book_index = books[books['title'] == selected_book].index[0]
    selected_features = user_item_matrix.iloc[:, book_index].values.reshape(1, -1)

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(user_item_matrix.T)
    distances, indices = knn.kneighbors(selected_features)

    recommended_books = books.iloc[indices[0]]['title'].tolist()
    recommended_images = books.iloc[indices[0]]['image_url'].tolist()

    # Plotting KNN distances
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(distances[0])), distances[0], color='skyblue')
    plt.yticks(range(len(distances[0])), recommended_books)
    plt.xlabel('Distance')
    plt.title('KNN Distances for Selected Book')
    st.pyplot(plt)

    return recommended_books, recommended_images

# Store conversation history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def add_to_chat(role, message):
    st.session_state.chat_history.append((role, message))

# Sidebar for navigation
selected_option = st.sidebar.selectbox("Choose an option",
    ["Top Rated Books","Book Recommendations", "Recommendations by User ID", "Predict Book Preference", "ROC Curve Comparison","User Review Sentiment Analysis"])

# Chat interface
def display_chat():
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div style='text-align: right; color: blue;'>**You:** {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: green;'>**Bot:** {message}</div>", unsafe_allow_html=True)
# User Review Sentiment Analysis

if selected_option == "User Review Sentiment Analysis":
    st.header("Analyze Your Review Sentiment")
    user_id = st.number_input("Enter Your User ID:", min_value=1, step=1)
    selected_book = st.selectbox("Select a Book:", book_name_list)
    user_review = st.text_area("Write your review here:")

    if st.button("Analyze Sentiment"):
        if user_review:
            analyzer.set_user_input(user_review)

            # Get the prediction
            result = analyzer.predict_sentiment()

            add_to_chat("user", f"Review: {user_review}")
            add_to_chat("bot", f"The sentiment of the review is: **{result}**")

        display_chat()

elif selected_option == "Top Rated Books":
    st.header("📚 Top 5 Rated Books")
    top_books = get_top_rated_books(books, ratings, n=5)

    for index, row in top_books.iterrows():
        col1, col2 = st.columns(2)
        with col1:
            st.image(row['image_url'], width=100)
        with col2:
            st.write(row['title'])

elif selected_option == "Book Recommendations":
    st.header('Get Book Recommendations')
    book_name = st.selectbox('Select Book Name', book_name_list)

    if st.button('Get Book Recommendations'):
        recommendations, images = get_knn_recommendations(book_name)
        add_to_chat("user", f"Get recommendations for '{book_name}'")
        response = f"Recommendations based on '{book_name}':"
        add_to_chat("bot", response)

        for i in range(len(recommendations)):
            col1, col2 = st.columns(2)
            with col1:
                st.image(images[i], width=100)
            with col2:
                st.write(recommendations[i])
        display_chat()

elif selected_option == "Recommendations by User ID":
    st.header('Get Recommendations by User ID')
    user_id_input = st.text_input('Enter your User ID:')

    if st.button('Get Recommendations'):
        try:
            user_id = int(user_id_input)
            if user_id in user_item_matrix.index:
                recommendations = get_recommendations_for_user(user_id)
                add_to_chat("user", f"Get recommendations for User ID {user_id}")
                response = f"Top 5 recommended books for User ID {user_id}:"
                add_to_chat("bot", response)

                for title, img in zip(recommendations['recommended_books'], recommendations['image_urls']):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, width=100)
                    with col2:
                        st.write(title)

                user_cluster = kmeans.labels[user_item_matrix.index.get_loc(user_id)]
                add_to_chat("bot", f"User ID {user_id} belongs to Cluster {user_cluster + 1}.")
                display_chat()

            else:
                add_to_chat("bot", "User ID not found. Please enter a valid User ID.")
                display_chat()
        except ValueError:
            add_to_chat("bot", "Please enter a valid integer for User ID.")
            display_chat()

elif selected_option == "Predict Book Preference":
    st.header('Predict Your Book Preference')
    selected_book_name = st.selectbox('Select a book to predict preference', book_name_list)
    user_id_input_class = st.text_input('Enter your User ID for prediction:')

    model_option = st.selectbox('Select a classification model', ['Random Forest', 'SVM', 'Decision Tree', 'Logistic Regression'])

    if st.button('Predict if you will like this book'):
        try:
            user_id = int(user_id_input_class)
            book_id = books.loc[books['title'] == selected_book_name, 'book_id'].values[0]
            model = {
                'Random Forest': clf_rf,
                'SVM': clf_svm,
                'Decision Tree': clf_hc_dt,
                'Logistic Regression': clf_hc_lr
            }[model_option]

            if user_id in features['user_id'].values and book_id in features['book_id'].values:
                prediction = predict_user_like(user_id, book_id, model)
                average_rating = books.loc[books['book_id'] == book_id, 'average_rating'].values[0]
                input_features = np.array([[user_id, book_id, average_rating]])

                if model_option == 'Logistic Regression':
                    predicted_class = clf_hc_lr.predict(input_features)[0]  # Prediction is binary, so we can use it directly
                    # Ue the predicted_class to form your response
                    if predicted_class == 1:
                        response = f"You will likely like '{selected_book_name}'! 📖"
                    else:
                        response = f"You will likely not like '{selected_book_name}'. 😕"
                else:
                    predicted_proba = model.predict_proba(input_features)[0][1]

                    if prediction == 1:
                        response = f"You will likely like '{selected_book_name}'! Probability: {predicted_proba:.2f} 📖"
                    else:
                        response = f"You will likely not like '{selected_book_name}'. Probability: {predicted_proba:.2f} 😕"
                add_to_chat("user", f"Predict preference for '{selected_book_name}' using {model_option}")
                add_to_chat("bot", response)

                # Plot the Decision Tree if selected
                if model_option == 'Decision Tree':
                    plt.figure(figsize=(12, 8))
                    clf_hc_dt.plot()  # Ensure you call the plot method from your custom class
                    st.pyplot(plt)

            else:
                add_to_chat("bot", "User ID or Book ID not found. Please check your input.")

            display_chat()

        except ValueError:
            add_to_chat("bot", "Please enter a valid integer for User ID.")
            display_chat()
        except IndexError:
            add_to_chat("bot", "The selected book does not exist. Please select a different book.")
            display_chat()

elif selected_option == "ROC Curve Comparison":
    st.header('ROC Curve Comparison of Classification Algorithms')

    # Create a dictionary of classifiers
    classifiers = {
        'Random Forest': clf_rf,
        'SVM': clf_svm,
        'Decision Tree': clf_hc_dt,

    }

    # Calculate ROC curves
    roc_curves = calculate_roc_curves(X_test, y_test, classifiers)

    # Plot ROC curves
    plot_roc_curves(roc_curves)
