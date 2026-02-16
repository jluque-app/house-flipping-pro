import { Routes, Route } from 'react-router-dom';
import Layout from './layout/Layout';
import LandingPage from './pages/LandingPage';
import ChatWidget from './components/ChatWidget';

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app/*" element={<Layout />} />
      </Routes>
      <ChatWidget />
    </>
  );
}

export default App;
