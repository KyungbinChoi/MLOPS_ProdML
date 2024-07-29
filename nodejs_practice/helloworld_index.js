
const express = require('express');
const http = require('http');
const app = express();
const server = http.createServer(app);

app.get('/', (req,res) => {
   res.send('welcome!!');
});

const port = 3000;
server.listen(port, ()=>console.log('just listen'));

