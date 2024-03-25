const video = document.getElementById('myVideo');
const chatBox = document.getElementById('chat-box');

function closeModal() {
    document.getElementById("myModal").style.display = "none";
}

function checkModal(message, audioDataUrl) {
    // Check the variable every 1 second (1000 milliseconds)
var intervalId = setInterval(function() {
    // Check if the variable is true
    if (document.getElementById("myModal").style.display == "none") {
        // Execute your function
        appendAlexMessage2(message, audioDataUrl);
        clearInterval(intervalId);
    }
}, 1000); // Interval set to 1000 milliseconds (1 second)
}

function appendAlexMessage2(message, audioDataUrl) {
    // Append the text message
    const messageElement = document.createElement('div');
    const labelText = document.createElement('span');
    labelText.className = "label-text";
    const messageText = document.createElement('span');

    labelText.innerText = `Alex: `;
    messageText.innerText = `${message}`;

    messageElement.className = "chatbot-message"
    messageElement.appendChild(labelText);
    messageElement.appendChild(messageText);
    chatBox.appendChild(messageElement);

    // Create and append the audio element
    const audioElement = new Audio(audioDataUrl);
    audioElement.controls = true;
    chatBox.appendChild(audioElement);
    audioElement.style.display = 'none'

    // Play the video and loop when the audio starts playing
    audioElement.addEventListener('play', function() {
        video.loop = true; // Ensure video loops
        video.play();
    });

    // Pause the video when the audio stops playing
    audioElement.addEventListener('ended', function() {
        video.currentTime = video.duration;
        video.pause();
    });

    audioElement.play();

    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
}

function appendAlexMessage(message, audioDataUrl) {
    // Append the text message
    const messageElement = document.createElement('div');
    const labelText = document.createElement('span');
    labelText.className = "label-text";
    const messageText = document.createElement('span');

    labelText.innerText = `Alex: `;
    messageText.innerText = `${message}`;

    messageElement.className = "chatbot-message"
    messageElement.appendChild(labelText);
    messageElement.appendChild(messageText);
    chatBox.appendChild(messageElement);

    // Create and append the audio element
    const audioElement = new Audio(audioDataUrl);
    audioElement.controls = true;
    chatBox.appendChild(audioElement);
    audioElement.style.display = 'none'

    // Play the video and loop when the audio starts playing
    audioElement.addEventListener('play', function() {
        video.loop = true; // Ensure video loops
        video.play();
    });

    // Pause the video when the audio stops playing
    audioElement.addEventListener('ended', function() {
        video.currentTime = video.duration;
        video.pause();
    });

    audioElement.play();

    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
}

function appendUserMessage(message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    const labelText = document.createElement('span');
    labelText.className = "label-text";
    const messageText = document.createElement('span');

    labelText.innerText = `You: `;
    messageText.innerText = `${message}`;

    messageElement.className = "user-message"
    messageElement.appendChild(labelText);
    messageElement.appendChild(messageText);

    
    chatBox.appendChild(messageElement);

    const ellipse = document.createElement('div');
    ellipse.className = "lds-ellipsis";
    ellipse.setAttribute('id', "lds-ellipsis")

    const l1 = document.createElement('div');
    const l2 = document.createElement('div');
    const l3 = document.createElement('div');

    ellipse.appendChild(l1)
    ellipse.appendChild(l2)
    ellipse.appendChild(l3)

    // const chatBox = document.getElementById('chat-box');
    chatBox.appendChild(messageElement);
    chatBox.appendChild(ellipse);
    // appendLoadingDots();

    chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
}

function appendLoadingDots() {
    // <div id="lds-ellipsis" class="lds-ellipsis"><div></div><div></div><div></div><div></div></div>
    const ellipse = document.createElement('div');
    ellipse.className = "lds-ellipsis";

    const l1 = document.createElement('div');
    const l2 = document.createElement('div');
    const l3 = document.createElement('div');

    ellipse.appendChild(l1)
    ellipse.appendChild(l2)
    ellipse.appendChild(l3)

    const chatBox = document.getElementById('chat-box');
    chatBox.appendChild(ellipse);
}

// JavaScript function to trigger when the user hits Enter after typing in the input field
document.getElementById("user-input").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
      // Call your function here
      sendMessage();
    }
  });

// Example of how you might call your API and use the response
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const userMessage = userInput.value;
    if (userMessage.trim() === '') return;

    appendUserMessage(userMessage);

    fetch('/api/chatbot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: userMessage})
    })
    .then(response => response.json())
    .then(data => {
        appendAlexMessage2(data.message, data.audio);
    })
    .catch(error => console.error('Error:', error))
    .finally(() => {
        // Remove loading indicator after response received
        const ellipse = document.getElementById('lds-ellipsis');
        ellipse.remove();
    });

    userInput.value = ''; // Clear input field after sending message
}

window.onload = function() {
    console.log("IN ON LOAD")
    let userMessage = "Introduce yourself as Alex, an AI virtual healthcare assistant. What are the kinds of questions you can answer? Keep your response short, less than 75 tokens."
    // Actions to be performed when the page is fully loaded
    const ellipse = document.createElement('div');
    ellipse.className = "lds-ellipsis";
    ellipse.setAttribute('id', "lds-ellipsis")

    const l1 = document.createElement('div');
    const l2 = document.createElement('div');
    const l3 = document.createElement('div');

    ellipse.appendChild(l1)
    ellipse.appendChild(l2)
    ellipse.appendChild(l3)

    // const chatBox = document.getElementById('chat-box');
    chatBox.appendChild(ellipse);

    fetch('/api/chatbot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: userMessage})
    })
    .then(response => response.json())
    .then(data => {
        checkModal(data.message, data.audio);
    })
    .catch(error => console.error('Error:', error))
    .finally(() => {
        // Remove loading indicator after response received
        const ellipse = document.getElementById('lds-ellipsis');
        ellipse.remove();
    });
  };
  