'use client';

import { useState } from 'react';
import Image from 'next/image';
import { ArrowDownUp, ArrowRight, Check, ChessKnight, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import showcase from '@/public/showcase.json';

type Policy = 'personal' | 'shared' | 'population';
type Example = (typeof showcase.examples)[number];
type Move = { uci: string; san: string; probability: number };

const pieceNames: Record<string, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };
const phases: Record<string, string> = { opening: 'Opening', middlegame: 'Middlegame', endgame: 'Endgame' };
const descriptions: Record<Policy, string> = {
  personal: 'Based on this player’s earlier games.',
  shared: 'An average of learned player preferences.',
  population: 'General move patterns without a player profile.',
};

function squareAt(index: number) {
  return 'abcdefgh'[index % 8] + (8 - Math.floor(index / 8));
}

function Chessboard({ example, highlighted, flipped }: { example: Example; highlighted: string; flipped: boolean }) {
  const squares: (string | null)[] = example.fen.split(' ')[0].split('/').flatMap(
    rank => rank.split('').flatMap(token => /\d/.test(token) ? Array.from({ length: Number(token) }, () => null) : [token]),
  );
  const indexes = Array.from({ length: 64 }, (_, index) => flipped ? 63 - index : index);

  return <figure className="board" aria-label={'Recorded ' + example.phase + ' position. ' + example.context.color + ' to move. Highlighted move ' + highlighted + '.'}>
    {indexes.map((index, displayIndex) => {
      const piece = squares[index];
      const square = squareAt(index);
      const light = (Math.floor(index / 8) + index % 8) % 2 === 0;
      const selected = square === highlighted.slice(0, 2) || square === highlighted.slice(2, 4);
      const pieceFile = piece ? 'Chess_' + piece.toLowerCase() + (piece === piece.toUpperCase() ? 'l' : 'd') + 't45.svg' : null;
      return <div key={square} className={['square', light ? 'light' : 'dark', selected ? 'highlighted' : ''].join(' ')}>
        {pieceFile && piece && <Image className="piece" src={'/chess-pieces/' + pieceFile} unoptimized
          alt={(piece === piece.toUpperCase() ? 'White ' : 'Black ') + pieceNames[piece.toLowerCase()] + ' on ' + square} width={45} height={45} />}
        {displayIndex % 8 === 0 && <span className="rank-coordinate" aria-hidden="true">{square[1]}</span>}
        {displayIndex >= 56 && <span className="file-coordinate" aria-hidden="true">{square[0]}</span>}
      </div>;
    })}
  </figure>;
}

export default function Home() {
  const [index, setIndex] = useState(0);
  const [policy, setPolicy] = useState<Policy>('personal');
  const [flipped, setFlipped] = useState(false);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  const example = showcase.examples[index];
  const moves: Move[] = example.policies[policy];
  const activeMove = highlighted ?? moves[0].uci;
  const actualRank = moves.findIndex(move => move.uci === example.actual.uci);
  const selectExample = (next: number) => { setIndex(next); setHighlighted(null); setRevealed(false); };
  const revealMove = () => { setRevealed(true); setHighlighted(example.actual.uci); };

  return <main className="app-shell">
    <header className="topbar">
      <a className="wordmark" href="#explorer"><ChessKnight strokeWidth={1.8} /><span>chess clone</span></a>
      <span className="topbar-note">Position explorer</span>
    </header>
    <section id="explorer" className="explorer" aria-label="Chess position explorer">
      <div className="board-column">
        <div className="position-heading">
          <div><span className="position-count">Position {index + 1} of {showcase.examples.length}</span><h1>{phases[example.phase]}</h1></div>
          <div className="player-context"><span className={'turn-indicator ' + example.context.color} />{example.context.color === 'white' ? 'White' : 'Black'} to move</div>
        </div>
        <Chessboard example={example} highlighted={activeMove} flipped={flipped} />
        <div className="board-toolbar">
          <Button variant="ghost" className="flip-button" onClick={() => setFlipped(!flipped)} aria-label="Flip board"><ArrowDownUp /> Flip board</Button>
          <div className="position-controls">
            <Button variant="ghost" size="icon" disabled={index === 0} onClick={() => selectExample(index - 1)} aria-label="Previous position"><ChevronLeft /></Button>
            <Button variant="ghost" size="icon" disabled={index === showcase.examples.length - 1} onClick={() => selectExample(index + 1)} aria-label="Next position"><ChevronRight /></Button>
          </div>
        </div>
        <nav className="example-tabs" aria-label="Choose a recorded position">{showcase.examples.map((item, itemIndex) => <Button key={item.id} variant="ghost"
          className={'example-tab ' + (itemIndex === index ? 'active' : '')} aria-pressed={itemIndex === index} onClick={() => selectExample(itemIndex)}>
          <span className="example-number">{itemIndex + 1}</span><span>{phases[item.phase]}</span>
        </Button>)}</nav>
      </div>
      <aside className="analysis-column" aria-label="Move predictions">
        <div className="analysis-intro"><h2>What would they play?</h2><p>Compare three predictions, then reveal the move from the game.</p></div>
        <Tabs value={policy} onValueChange={value => { setPolicy(value as Policy); setHighlighted(null); }}>
          <TabsList className="policy-tabs" aria-label="Prediction type"><TabsTrigger value="personal">Player</TabsTrigger><TabsTrigger value="shared">Shared</TabsTrigger><TabsTrigger value="population">General</TabsTrigger></TabsList>
        </Tabs>
        <p className="policy-description">{descriptions[policy]}</p>
        <div className="list-heading"><span>Likely moves</span><span>Chance</span></div>
        <ol className="move-list">{moves.map((move, moveIndex) => <li key={move.uci}>
          <Button variant="ghost" className={'move-choice ' + (activeMove === move.uci ? 'selected' : '')} aria-pressed={activeMove === move.uci}
            onClick={() => setHighlighted(move.uci)} aria-label={move.san + ', ' + (move.probability * 100).toFixed(1) + ' percent. Highlight on board.'}>
            <span className="move-rank">{moveIndex + 1}</span><span className="move-notation">{move.san}<small>{move.uci.slice(0, 2)} <ArrowRight /> {move.uci.slice(2, 4)}</small></span>
            <span className="probability-cell"><strong>{(move.probability * 100).toFixed(1)}<span>%</span></strong><span className="probability-track"><span style={{ width: move.probability * 100 + '%' }} /></span></span>
            <span className="move-indicator">{revealed && move.uci === example.actual.uci ? <Check aria-label="Recorded move" /> : <ChevronRight />}</span>
          </Button>
        </li>)}</ol>
        {!revealed ? <Button className="reveal-button" onClick={revealMove}>Reveal the move played <ArrowRight /></Button> :
          <Button variant="ghost" className="actual-move" onClick={() => setHighlighted(example.actual.uci)} aria-label={'Highlight the recorded move ' + example.actual.san}>
            <span><small>Move played</small><strong>{example.actual.san}</strong></span>
            <span className={'match-state ' + (actualRank >= 0 ? 'match' : 'miss')}>{actualRank >= 0 ? <Check /> : <ArrowRight />}{actualRank === 0 ? 'Top guess matched' : actualRank > 0 ? `Ranked #${actualRank + 1}` : 'Outside top three'}</span>
          </Button>}
        <p className="app-note">Curated saved positions, not a representative sample. This preview does not make live predictions.</p>
      </aside>
    </section>
  </main>;
}
