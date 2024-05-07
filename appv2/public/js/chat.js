// --- chat message web component
class ChatMessage extends HTMLElement {
  constructor({ ts, chatrole, content, links } = { ts: new Date(), chatrole: "assistant", content: "", links: [] }) {
    super()

    this.ts = ts ?? new Date();
    this.chatrole = chatrole ?? "assistant";
    this.content = content ?? "";
    this.links = links ?? [];
  }

  // --- attributes
  static get observedAttributes() {
    return ['ts', 'content', 'chatrole', 'links'];
  }

  // --- events
  connectedCallback() {
    this.render()
  }

  attributeChangedCallback(property, oldValue, newValue) {
    if (oldValue === newValue) return;
    this[property] = newValue;
    this.render();
  }

  // --- element HTML
  render() {
    const imageUrl = this.chatrole === "assistant" ? "/assets/img/assistant.png" : "/assets/img/user.png";
    const details = (link) => {
      if (link.address === "") return "";
      return `
        ${link.address}<br />
        ${(link.price * 1000000).toLocaleString('id-ID', {style: 'currency', currency: 'IDR'})}
      `
    }

    this.innerHTML = `
      <div class="message-item-box">
        <article class="media">
          <div class="media-left">
            <figure class="image is-64x64">
              <img class="recolor-avatar" src="${imageUrl}" alt="Image" />
            </figure>
          </div>
          <div class="media-content">
            <div class="content">
              <div>
                <small>${this.ts.toLocaleString('id-ID')}</small><br />
                <p>${this.content}</p>
              </div>
            </div>

            ${this.links.map(link => `
              <a href="${link.url}" target="_blank" class="button is-small mr-4">
                <div class="flex is-flex-direction-column">
                  <img class="image is-4by3 house-image mb-2" src="${link.image_url}">
                  ${details(link)}
                </div>
              </a>
            `).join('\n')}
          </div>
        </article>
      </div>
      `
  }
}

window.customElements.define('chat-message', ChatMessage);

// --- event listener
/** @type {HTMLButtonElement} */
const sendButton = document.getElementById("send-button");
/** @type {HTMLInputElement} */
const uploadButton = document.getElementById("upload-button");
/** @type {HTMLTextAreaElement} */
const chatBoxTextarea = document.getElementById("chat-box");
/** @type {HTMLDivElement} */
const messagesContainer = document.getElementById('messages-container');

const sessionId = crypto.randomUUID();

function send() {
  // check if input is empty
  if (chatBoxTextarea.value === "" && uploadButton.files.length === 0) return;
  sendButton.setAttribute("disabled", "disabled");
  uploadButton.setAttribute("disabled", "disabled");
  chatBoxTextarea.setAttribute("disabled", "disabled");

  // build message
  const data = new FormData();
  data.append("session_token", sessionId);

  // upload file or prompt
  if (uploadButton.files.length > 0) {
    data.append("file", uploadButton.files[0]);
  } else {
    data.append("prompt", chatBoxTextarea.value);
  }

  // add a new message
  if (uploadButton.files.length > 0) {
    const imageObject = uploadButton.files[0];
    const imageBlob = new Blob([imageObject], { type: imageObject.type });

    messagesContainer.appendChild(new ChatMessage({
      ts: new Date(),
      chatrole: "user",
      content: "Pencarian menggunakan gambar",
      links: [{
        address: "",
        price: "",
        image_url: URL.createObjectURL(imageBlob),
        url: "#"
      }],
    })).scrollIntoView({ behavior: "smooth" });
  } else {
    messagesContainer.appendChild(new ChatMessage({
      ts: new Date(),
      chatrole: "user",
      content: chatBoxTextarea.value,
      links: [],
    })).scrollIntoView({ behavior: "smooth" });
  }
  
  // call completion API
  fetch("/chat/completion", {
    method: "POST",
    body: data,
  })
  .then(response => response.json())
  .then(data => {
    // clear chat box
    chatBoxTextarea.value = "";

    // clear file upload
    uploadButton.value = null;

    // append the reply
    messagesContainer.appendChild(new ChatMessage({
      ts: new Date(),
      chatrole: "assistant",
      content: data.content,
      links: data.links,
    })).scrollIntoView({ behavior: "smooth" });
  })
  .catch(err => {
    // append the reply
    messagesContainer.appendChild(new ChatMessage({
      ts: new Date(),
      chatrole: "assistant",
      content: "Gagal menghubungi server. Coba beberapa saat lagi.",
      links: [],
    })).scrollIntoView({ behavior: "smooth" });
  })
  .finally(() => {
    // enable controls
    sendButton.removeAttribute("disabled");
    uploadButton.removeAttribute("disabled");
    chatBoxTextarea.removeAttribute("disabled");

    // always clear file upload
    uploadButton.value = null;
  })
}

chatBoxTextarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    send();
  }
});

uploadButton.addEventListener("change", (e)=>{
  e.preventDefault();
  send();
})

sendButton.addEventListener('click', (e) => {
  e.preventDefault();
  send();
});