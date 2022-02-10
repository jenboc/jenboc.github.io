words = ["tick", "tock"]

function ClearContainer(){
	var div = document.getElementById('random-container');
	div.innerHTML = "";
	localStorage.ticktock = div.innerHTML
}

function randNumber(max){
	return Math.floor(Math.random() * max)
}


function RandHex(){
	red = randNumber(255);
	green = randNumber(255);
	blue = randNumber(255);
	
	return red.toString(16) + green.toString(16) + blue.toString(16);
}


function AddRandom(){
	var div = document.getElementById('random-container')
	
	randWord = words[randNumber(words.length)]	

	var randX = randNumber(105);
	var randY = randNumber(105);
	var randDeg = randNumber(360);
	
	var randColor = "";
	var hex = ["A", "B", "C", "D", "E", "F"]
	
	for (let i = 0; i < 6; i++){
		let randByte = randNumber(255)
		var randHex = hex[255%16] + hex[Math.floor(255/16)]
		
		randColor += randHex
	}	

	var newDiv = document.createElement('div')
	newDiv.innerText = randWord
	newDiv.className = 'random-word'
	newDiv.style.cssText = "color: " + RandHex() + ";right:"+ randX + "%;top:" + randY + "%;transform: rotate("+ randDeg + "deg);"
	div.appendChild(newDiv)
	localStorage.ticktock = div.innerHTML
}

function SetTime(){
	var div = document.getElementById('time');
	let date = new Date();
	
	let hours = date.getHours();
	let mins = date.getMinutes();
	let secs = date.getSeconds();

	if (hours.toString().length == 1){
		hours = "0" + hours
	}
	
	if (mins.toString().length == 1){
		mins = "0" + mins
	}

	if (secs.toString().length == 1){
		secs = "0" + secs
	}

	div.innerText = hours + ":" + mins + ":" + secs;
}

function ChangeClockColor(){
	bgColor = RandHex();
	fgColor = RandHex();
	div = document.getElementById('time')
	div.style.cssText = "background-color: " + bgColor + ";color: " + fgColor + ";"
}

container = document.getElementById('random-container')
container.innerHTML = localStorage.ticktock

var query = window.location.search;
var parameters = new URLSearchParams(query);

var cchange = 0;
if (parameters.has('cchange')){
	cchange = parameters.get('cchange')
}
else{
	cchange = 500
}

var tadd = 0;
if (parameters.has('tadd')){
	tadd = parameters.get('tadd')
}
else {
	tadd = 1000
}

window.setInterval(SetTime, 1000);
window.setInterval(ChangeClockColor,cchange);
window.setInterval(AddRandom, tadd);