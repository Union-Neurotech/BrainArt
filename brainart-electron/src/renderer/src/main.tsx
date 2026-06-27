import { createRoot } from 'react-dom/client'
import { setupIonicReact } from '@ionic/react'
import App from './App'

// Ionic core styles + forced dark palette
import '@ionic/react/css/core.css'
import '@ionic/react/css/normalize.css'
import '@ionic/react/css/structure.css'
import '@ionic/react/css/typography.css'
import '@ionic/react/css/palettes/dark.always.css'

import './index.css'

setupIonicReact()

createRoot(document.getElementById('root')!).render(<App />)
