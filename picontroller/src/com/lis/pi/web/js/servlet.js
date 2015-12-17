var ws;
function init() {
	ws = new WebSocket("ws://" + window.location.host + "/info");
	ws.onopen = function() {
		console.log("Opened connection");
	};
	ws.onmessage = function(evt) {
		var msg = evt.data;
		console.log(msg);
		var data = JSON.parse(msg);
		if (data instanceof Array) {
			
		}
		var type = data.type;
		if (type == 'MEMORY') {
			var memoryBar = document.getElementById("memoryBar");
			memoryBar.style.width = data.value * 100 + '%';
		} else if (type == 'LIGHT') {
			var lightBar = document.getElementById("lightBar");
			lightBar.style.width = data.value / 2 * 100 + '%';
		} else if (type == 'TEMPERATURE') {
			var heatBar = document.getElementById("heatBar");
			heatBar.style.width = data.value + '%';
		}
	};
	ws.onclose = function() {
		console.log("Closed connection");
		setTimeout(init, 5000);
	};
}