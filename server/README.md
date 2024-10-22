To setup the backend server:
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask server:
```bash
python3 app.py
```

3. In your react/react-native app, set the proper server IP address & domain for api server and send POST request, example usage:
```tsx
// code to test Python backend server
  const name = "Patrick"
  const [responseData, setResponseData] = useState('');
  const [error, setError] = useState('');
  const API_URL = 'http://your-server-ip/api/process';
  const sendData = async () => {
    try {
      console.log('Sending request to:', API_URL);
      const response = await axios.post(API_URL, {
        name: name
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      console.log('Response:', response.data);
      setResponseData(response.data.message);
      setError('');
    } catch (error) {
      console.log('Error details:', error.response || error);
      setError(error.response?.data?.error || error.message);
      setResponseData('');
    }
  };
```