import http
import random
import json
import torch
from models import NeuralNetwork
from nltk_utilities import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # to check for gpu is avaiable otherwise we use the cpu

with open ('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load (FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device) # push the model to device if it is available 
model.load_state_dict(model_state)  # now it knows our learned parameters
model.eval()    # set to evaluation mode

bot_name = "Mabi"
print("Let's Chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    sentence =  tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0]) # 0 column and 1 rows shape of input text
    X = torch.from_numpy(X)  # then we convert it because our bag of words return a numpy array

    # prediction block
    output = model(X)
    _, predicted = torch.max(output, dim=1) # dim=> dimension
    tag = tags [predicted.item()]   # find the tag in the intents.json file

    #applying the softmax to get the actual probabilities

    probabilities = torch.softmax(output, dim=1)   # inspecting the softmax of the raw output, and setting dimension to 1
    probability = probabilities[0][predicted.item()]

    if probability.item() > 0.75: # if the probability is high enough perform task
    # finding the corresponding intent by looping over intents to see if the tags matches
        for intent in intents ["intents"]:
         if tag == intent["tag"]:
            print(f"{bot_name}: {random.choice(intent['responses'])}")  # this will print the reponse of the bot
    else:       # if the probabilty is not high enough
        print(f"{bot_name}: I do not understand...")
        # print(f"(bot_name): {choice(intent['noanswer']}")

# chatbot.py

class ChatbotHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/get-response'):
            # Extract the user's message from the URL query parameter
            user_message = self.path.split('=')[-1]

            # Your chatbot logic starts here
            sentence = tokenize(user_message)
            X = bag_of_words(sentence, all_words)  # Replace 'all_words' with your actual variable
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X)

            # Prediction block
            output = model(X)  # Replace 'model' with your actual model
            _, predicted = torch.max(output, dim=1)
            tag = tags[predicted.item()]

            probabilities = torch.softmax(output, dim=1)
            probability = probabilities[0][predicted.item()]

            if probability.item() > 0.75:
                for intent in intents["intents"]:
                    if tag == intent["tag"]:
                        response = random.choice(intent['responses'])
                        break
                else:
                    response = "I do not understand..."
            else:
                response = "I do not understand..."
            # Your chatbot logic ends here

            # Send a response back to the client
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(response.encode())
        else:
            # Handle other requests (e.g., serving static files)
            super().do_GET()

if __name__ == "__main__":
    PORT = 8000
    server = http.server.HTTPServer(("", PORT), ChatbotHandler)
    print(f"Chatbot server running on port {PORT}")
    server.serve_forever()
