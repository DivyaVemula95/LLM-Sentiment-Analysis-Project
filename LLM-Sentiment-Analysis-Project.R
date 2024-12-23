# Install the required packages if not already installed
if (!require('reticulate')) install.packages('reticulate'); library('reticulate')
if (!require('torch')) install.packages('torch'); library('torch')
if (!require('text')) install.packages('text'); library('text')
if (!require('dplyr')) install.packages('dplyr'); library('dplyr')


# Specify the path to your virtual environment
reticulate::use_virtualenv("/Users/divyavemula/.virtualenvs/r-reticulate")
py_config()
# Use reticulate to install and use Python libraries (if not installed)
reticulate::py_install("datasets")
# Load Python libraries using reticulate
transformers <- import("transformers")
torch <- import("torch")

# Load the pre-trained sentiment analysis model
# Load the pre-trained model and tokenizer for sentiment analysis
model_name <- "distilbert-base-uncased-finetuned-sst-2-english"
model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_name)
tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)

# Function to perform sentiment analysis
sentiment_analysis <- function(text) {
  # Tokenize the input text using the correct tokenizer method
  inputs <- tokenizer(text, padding=TRUE, truncation=TRUE, max_length=512L, return_tensors="pt")  # Ensure max_length is an integer
  
  # Perform the forward pass through the model to get predictions
  with(torch$no_grad(), {
    logits <- model$forward(inputs$input_ids)
  })
  
  # Check the shape of the logits to ensure it is correct
  print(logits$logits$shape)  # Debugging the shape of the logits
  
  # Get the predicted class (0 for negative, 1 for positive)
  predictions <- torch$argmax(logits$logits, dim=1L)  # Ensure dim is passed as integer (1L)
  
  return(predictions)
}

# Example text for sentiment analysis
text <- "I love this product!"

# Perform sentiment analysis
prediction <- sentiment_analysis(text)

# Print the result
print(prediction)

# Extract the value from the tensor for comparison
prediction_value <- prediction$item()  # .item() converts the tensor to an R scalar

# Return sentiment result based on prediction
if (prediction_value == 1) {
  print("Positive sentiment")
} else {
  print("Negative sentiment")
}


# Test the function with multiple texts
# Example texts for sentiment analysis
texts <- c("I love this movie!", "This is the worst experience ever.")

# Loop through each text and get sentiment analysis result
for (text in texts) {
  result <- sentiment_analysis(text)  # Use the correct function name
  # Extract the value from the tensor for comparison
  prediction_value <- result$item()  # .item() converts the tensor to a scalar
  
  # Print sentiment result based on prediction
  sentiment <- if (prediction_value == 1) {
    "Positive sentiment"
  } else {
    "Negative sentiment"
  }
  
  print(paste("Text:", text, "=> Sentiment:", sentiment))
}



# Load your own dataset (e.g., a CSV with text and labels)
# Use a DataFrame with 'text' and 'label' columns
# Ensure max_length is passed as an integer

# Ensure tokenizer is correctly padding the inputs
tokenizer1 <- transformers$AutoTokenizer$from_pretrained("distilbert-base-uncased")

# Example dataset
your_data <- data.frame(
  text = c("I love this movie!", "This is the worst experience ever."),
  label = c(1, 0)  # 1 for positive, 0 for negative
)

# Define a custom dataset to work with
CustomDataset <- torch::nn_module(
  "CustomDataset",
  
  initialize = function(data, tokenizer) {
    self$data <- data
    self$tokenizer <- tokenizer
  },
  
  forward = function(index) {
    # Extract the text and label at the given index
    text <- self$data$text[index]
    label <- self$data$label[index]
    
    # Tokenize the text with padding and truncation, ensuring uniform length
    inputs <- self$tokenizer(text, padding = "max_length", truncation = TRUE, max_length = as.integer(512), return_tensors = "pt")
    
    return(list(inputs$input_ids, label))
  }
)

# Instantiate the dataset with your data and tokenizer
train_dataset <- CustomDataset$new(data = your_data, tokenizer = tokenizer1)

# Define batch size
batch_size <- 2

create_batches <- function(dataset, batch_size) {
  num_samples <- length(dataset$data$text)
  batches <- list()
  
  for (i in seq(1, num_samples, by = batch_size)) {
    batch_indices <- i:min(i + batch_size - 1, num_samples)
    batch_data <- lapply(batch_indices, function(idx) dataset$forward(idx))
    
    # Ensure inputs are tensors, and explicitly cast dim to integer (as required)
    inputs <- torch$stack(lapply(batch_data, function(x) x[[1]]), dim = as.integer(0))  # Stack tensors along dim=0
    
    # Convert labels to tensor
    labels <- torch$tensor(sapply(batch_data, function(x) x[[2]]))
    
    batches[[length(batches) + 1]] <- list(inputs, labels)
  }
  
  return(batches) 
}

# Create batches from the dataset
train_batches <- create_batches(train_dataset, batch_size)

# Check the shape of the inputs in the first batch
print(train_batches[[1]])  # Check what the first batch contains
print(class(train_batches[[1]][[1]]))  # Should return "torch_tensor"
print(train_batches[[1]][[1]]$size())  # Get size of the first tensor in the batch

inputs <- train_batches[[1]][[1]]$squeeze()  # Automatically remove all dimensions of size 1
print(inputs$size())  # Should now return torch.Size([2, 512])

# Define the path to save the model and tokenizer
save_path <- file.path(getwd(), "fine_tuned_model")

# Save the fine-tuned model
model$save_pretrained(save_path)

# Save the tokenizer
tokenizer$save_pretrained(save_path)

# Print the saved location
print(paste("Model and tokenizer saved at: ", save_path))
C 