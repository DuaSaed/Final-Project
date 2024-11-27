from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('USA_cars_datasets.csv')
df.drop(columns=['Unnamed: 0', 'vin', 'lot', 'country', 'condition'], inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
df = pd.get_dummies(df, columns=['brand', 'title_status', 'color', 'state'], drop_first=True)

# Function to recommend cars based on user input
def recommend_cars(budget, min_year, max_mileage):
    filtered_df = df[(df['price'] <= budget) & (df['year'] >= min_year) & (df['mileage'] <= max_mileage)]
    
    if filtered_df.empty:
        print("No cars match your criteria.")

    features = filtered_df.drop(columns=['model', 'price', 'year', 'mileage'])
    similarity_matrix = cosine_similarity(features)
    top_indices = similarity_matrix[0].argsort()[::-1][1:6]
    recommendations = filtered_df.iloc[top_indices][['model', 'price', 'year', 'mileage']].reset_index(drop=True)
    return recommendations.to_dict(orient='records')

@app.route('/')
def home():
    # HTML code directly in the route
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Car Recommendation Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 30px; }
            #chatbox { border: 1px solid #ddd; padding: 15px; height: 400px; overflow-y: auto; margin-bottom: 10px; }
            .user-message { color: blue; }
            .bot-message { color: green; }
            #user_input { width: 70%; padding: 8px; }
            button { padding: 8px; margin-left: 5px; }
        </style>
    </head>
    <body>
        <h2>Car Recommendation Chatbot</h2>
        <div id="chatbox"></div>
        <input type="text" id="user_input" placeholder="Enter budget, year, mileage (e.g., 20000, 2015, 60000)">
        <button onclick="sendMessage()">Send</button>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
            function sendMessage() {
                const userInput = $('#user_input').val();
                const [budget, year, mileage] = userInput.split(',').map(Number);

                $('#chatbox').append(`<p class="user-message">You: ${userInput}</p>`);
                $('#user_input').val('');

                $.ajax({
                    url: '/get_recommendations',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ budget, year, mileage }),
                    success: function(response) {
                        if (typeof response === 'string') {
                            $('#chatbox').append(`<p class="bot-message">Bot: ${response}</p>`);
                        } else {
                            response.forEach(car => {
                                $('#chatbox').append(
                                    `<p class="bot-message">Model: ${car.model}, Price: $${car.price}, Year: ${car.year}, Mileage: ${car.mileage}</p>`
                                );
                            });
                        }
                    }
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    user_input = request.json
    budget = int(user_input.get('budget'))
    min_year = int(user_input.get('year'))
    max_mileage = int(user_input.get('mileage'))
    
    recommendations = recommend_cars(budget, min_year, max_mileage)
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
# Open your browser and go to:
# http://localhost:5000 
# for working interface of chatbot