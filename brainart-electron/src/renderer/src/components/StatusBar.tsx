import {
  IonButton,
  IonInput,
  IonProgressBar,
  IonRange,
  IonSelect,
  IonSelectOption
} from '@ionic/react'
import type { Board } from '../ws'
import type { BrainState } from '../state'

interface Props {
  boards: Board[]
  selectedBoard: string
  onSelectBoard: (name: string) => void
  port: string
  onPort: (p: string) => void
  online: boolean
  connected: boolean
  streaming: boolean
  device: string | null
  onConnect: () => void
  onDisconnect: () => void
  onStart: () => void
  onStop: () => void
  ui: BrainState
  onSlider: (key: keyof BrainState, value: number) => void
  onSave: () => void
  onPrint: () => void
}

function Indicator({ label, value }: { label: string; value: number }) {
  return (
    <div className="indicator">
      <div className="indicator-head">
        <span className="indicator-label">{label}</span>
        <span className="indicator-value">{value.toFixed(2)}</span>
      </div>
      <IonProgressBar value={Math.max(0, Math.min(1, value))} />
    </div>
  )
}

function Slider({
  label,
  hint,
  min,
  max,
  step,
  value,
  onChange
}: {
  label: string
  hint: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
}) {
  return (
    <div className="slider-row">
      <div className="indicator-head">
        <span className="indicator-label">
          {label} <small>{hint}</small>
        </span>
        <span className="indicator-value">{step >= 1 ? value.toFixed(0) : value.toFixed(2)}</span>
      </div>
      <IonRange
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onIonInput={(e) => onChange(e.detail.value as number)}
      />
    </div>
  )
}

export default function StatusBar(props: Props) {
  const {
    boards,
    selectedBoard,
    onSelectBoard,
    port,
    onPort,
    online,
    connected,
    streaming,
    device,
    onConnect,
    onDisconnect,
    onStart,
    onStop,
    ui,
    onSlider,
    onSave,
    onPrint
  } = props

  const usingPort = boards.find((b) => b.name === selectedBoard)?.using_port ?? false

  return (
    <div className="statusbar">
      <div className="sb-header">
        <h1>BrainArt</h1>
        <p>{online ? (connected ? `device: ${device}` : 'backend ready') : 'connecting to backend…'}</p>
      </div>

      <div className="sb-scroll">
        {/* ── Device / connection ── */}
        <section className="sb-sec">
          <div className="sb-sec-title">Device</div>
          <IonSelect
            aria-label="Board"
            interface="popover"
            value={selectedBoard}
            disabled={connected}
            placeholder="Select a board"
            onIonChange={(e) => onSelectBoard(e.detail.value)}
          >
            {boards.map((b) => (
              <IonSelectOption key={b.name} value={b.name}>
                {b.name}
              </IonSelectOption>
            ))}
          </IonSelect>

          {usingPort && (
            <IonInput
              className="sb-port"
              label="Port"
              labelPlacement="stacked"
              placeholder="e.g. COM3"
              value={port}
              disabled={connected}
              onIonInput={(e) => onPort(e.detail.value ?? '')}
            />
          )}

          {!connected ? (
            <IonButton expand="block" disabled={!online || !selectedBoard} onClick={onConnect}>
              Connect
            </IonButton>
          ) : (
            <IonButton expand="block" color="danger" onClick={onDisconnect}>
              Disconnect
            </IonButton>
          )}

          <div className="btn-pair">
            <IonButton
              expand="block"
              color="success"
              disabled={!connected || streaming}
              onClick={onStart}
            >
              Start
            </IonButton>
            <IonButton expand="block" color="medium" disabled={!streaming} onClick={onStop}>
              Stop
            </IonButton>
          </div>
        </section>

        {/* ── Live / measured metrics ── */}
        <section className="sb-sec">
          <div className="sb-sec-title">ML Metrics</div>
          <Indicator label="Concentration" value={ui.concentration} />
          <Indicator label="Relaxation" value={ui.relaxation} />
          <Indicator label="Meditative state" value={ui.mindfulness} />
        </section>

        <section className="sb-sec">
          <div className="sb-sec-title">EEG Bands</div>
          <Indicator label="Delta (δ)" value={ui.delta} />
          <Indicator label="Theta (θ)" value={ui.theta} />
          <Indicator label="Alpha (α)" value={ui.alpha} />
          <Indicator label="Beta (β)" value={ui.beta} />
          <Indicator label="Gamma (γ)" value={ui.gamma} />
        </section>

        {/* ── Emotion (manual sliders this iteration) ── */}
        <section className="sb-sec">
          <div className="sb-sec-title">Emotion</div>
          <Slider
            label="Arousal"
            hint="zoom + structure"
            min={-1}
            max={1}
            step={0.01}
            value={ui.arousal}
            onChange={(v) => onSlider('arousal', v)}
          />
          <Slider
            label="Valence"
            hint="warm ↔ cool"
            min={-1}
            max={1}
            step={0.01}
            value={ui.valence}
            onChange={(v) => onSlider('valence', v)}
          />
          <Slider
            label="Depth"
            hint="network layers"
            min={1}
            max={12}
            step={1}
            value={ui.layers}
            onChange={(v) => onSlider('layers', v)}
          />
        </section>
      </div>

      <div className="sb-footer">
        <IonButton expand="block" fill="outline" onClick={onSave}>
          Save Image
        </IonButton>
        <IonButton expand="block" fill="outline" onClick={onPrint}>
          Print Image
        </IonButton>
      </div>
    </div>
  )
}
