function updateBitcoinPrice() {
    fetch('/live_price')
       .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
       .then(data => {
            const priceElement = document.querySelector('.price_live');
            if (priceElement) {
                priceElement.textContent = `current Price: $${data.price}`;
            }
        })
       .catch(error => {
            console.error('Error fetching Bitcoin price:', error);
        });
}

setInterval(updateBitcoinPrice, 10000);

// Call it once on page load to display initial price
updateBitcoinPrice();










