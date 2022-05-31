// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sim

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

// CorpusEnv reads text from a file and presents a window of sequential words
// as the input.  Words included in the vocabulary can be filtered by frequency
// at both ends.
// For input, a simple bag-of-words is used, with words encoded as localist 1-hot
// units (requiring a large input layer) or using random distributed vectors in a
// lower-dimensional space.
type CorpusEnv struct {
	Nm          string             `desc:"name of this environment"`
	Dsc         string             `desc:"description of this environment"`
	Words       []string           `desc:"full list of words used for activating state units according to index"`
	WordMap     map[string]int     `desc:"map of words onto index in Words list"`
	FreqMap     map[string]float64 `desc:"map of words onto frequency in entire corpus, normalized"`
	Corpus      []string           `desc:"entire corpus as one long list of words"`
	Sentences   [][]string         `desc:"full list of sentences"`
	SentOffs    []int              `desc:"offsets into corpus for each sentence"`
	NGrams      NGramMap           `desc:"normalized frequency of a word given n-words-1 of context"`
	WordReps    etensor.Float32    `desc:"map of words into random distributed vector encodings"`
	CurWords    []string           `desc:"The current words of context"`
	CurNextWord string             `desc:"The current successor word"`
	Input       etensor.Float32    `desc:"current window activation state"`
	Output      etensor.Float32    `desc:"successor word target activation state"`

	NContext       int `desc:"number of words in context (ngram -1)"`
	NSuccessor     int `desc:"max number of successors in the ngram map"`
	NRandomizeWord int `desc:"Every this many ticks, reseed the current word"`

	Localist   bool       `desc:"use localist 1-hot encoding of words -- else random dist vectors"`
	DistPctAct float64    `desc:"distributed representations of words: target percent activity total for WindowSize words all on at same time"`
	DropOut    bool       `desc:"randomly drop out the highest-frequency inputs"`
	UseUNK     bool       `desc:"use an UNK token for unknown words -- otherwise just skip"`
	InputSize  evec.Vec2i `desc:"size of input layer state"`
	MaxVocab   int        `desc:"maximum number of words representable -- InputSize.X*Y"`
	VocabFile  string     `desc:"location of the generated vocabulary file"`
	CorpStart  int        `inactive:"+" desc:"for this processor (MPI), starting index into Corpus"`
	CorpEnd    int        `inactive:"+" desc:"for this processor (MPI), ending index into Corpus"`

	Run   env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch env.Ctr `view:"inline" desc:"epoch is arbitrary increment of number of times through trial.Max steps"`
	Trial env.Ctr `view:"inline" desc:"trial is the network training step counter"`
	Tick  env.Ctr `view:"inline" desc:"tick counts steps through the Corpus"`
	Block env.Ctr `view:"inline" desc:"block counts iterations through the entire Corpus"`
}

// NGramMap stores a normalized frequency of a word given n-words-1 of context
// First key is context which may be multiple words space separated,
// second map is successor words with frequency
type NGramMap map[string]map[string]float64

func (ngm *NGramMap) Add(context, successor string) {
	freqmap, has := (*ngm)[context]
	if !has {
		freqmap = make(map[string]float64)
	}
	freq := freqmap[successor]
	freqmap[successor] = freq + 1
	(*ngm)[context] = freqmap
}

func (ngm *NGramMap) TopNSuccessors(n int) {
	var freqs []float64

	for _, freqmap := range *ngm {
		if cap(freqs) < len(freqmap) {
			freqs = make([]float64, len(freqmap))
		} else {
			freqs = freqs[:len(freqmap)]
		}
		idx := 0
		for _, freq := range freqmap {
			freqs[idx] = freq
			idx++
		}

		sort.Sort(sort.Reverse(sort.Float64Slice(freqs)))
		var threshold float64 = 0
		if len(freqs) > n {
			threshold = freqs[n]
		}
		for successor, freq := range freqmap {
			if freq < threshold {
				delete(freqmap, successor)
			}
		}
	}
}

func (ngm *NGramMap) Normalize() {
	for _, freqmap := range *ngm {
		sum := 0.0
		for _, freq := range freqmap {
			sum += freq
		}
		for successor, freq := range freqmap {
			freqmap[successor] = freq / sum
		}
	}
}

var epsilon = 1e-7

func (ev *CorpusEnv) Name() string { return ev.Nm }
func (ev *CorpusEnv) Desc() string { return ev.Dsc }

// Validate initializes matrix and labels to given size
func (ev *CorpusEnv) Validate() error {
	return nil
}

func (ev *CorpusEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "Output":
		return &ev.Output
	}
	return nil
}

func (ev *CorpusEnv) Init(run int) {
	ev.Run.Scale = env.Run     //a full training of the network scale
	ev.Epoch.Scale = env.Epoch //a set of trials
	ev.Trial.Scale = env.Trial //a single example/ row/ state
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = 0 // init state -- key so that first Step() = 0
}

func (ev *CorpusEnv) Config(inputfile string, inputsize evec.Vec2i, localist bool, ncontext, ntopsuccessors, nrandomizeword int) {
	ev.InputSize = inputsize //the shape of the tensor
	ev.Localist = localist
	//ev.MaxVocab = ev.InputSize.X * ev.InputSize.Y
	ev.MaxVocab = 25 //TODO: make this logical
	ev.UseUNK = false
	ev.VocabFile = "stored_vocab_cbt_train.json" //subset of words
	ev.NContext = ncontext
	ev.NSuccessor = ntopsuccessors
	ev.NRandomizeWord = nrandomizeword

	ev.LoadFmFile(inputfile) // load the corpus

	ev.Input.SetShape([]int{ev.InputSize.Y, ev.InputSize.X}, nil, []string{"Y", "X"})
	ev.Output.SetShape([]int{ev.InputSize.Y, ev.InputSize.X}, nil, []string{"Y", "X"})
	ev.CurWords = make([]string, ev.NContext)

	ev.ConfigWordReps() //pattern for each word

}

// JsonData is the full original corpus, in sentence form
type JsonData struct {
	Vocab     []string
	Freqs     []int
	Sentences [][]string
}

// JsonVocab is the map-based encoding of just the vocabulary and frequency
type JsonVocab struct {
	Vocab map[string]int
	Freqs map[string]float64
}

func (ev *CorpusEnv) NormalizeFreqs() {
	var s float64
	s = 0
	for w, _ := range ev.FreqMap {
		// ev.FreqMap[w] = math.Sqrt(f)
		s += ev.FreqMap[w]
	}
	for w, _ := range ev.FreqMap {
		ev.FreqMap[w] /= s
	}
}

func (ev *CorpusEnv) LimitVocabulary() {
	freqs := make([]float64, len(ev.FreqMap))
	idx := 0
	for _, freq := range ev.FreqMap {
		freqs[idx] = freq
		idx++
	}

	sort.Sort(sort.Reverse(sort.Float64Slice(freqs)))

	if len(freqs) <= ev.MaxVocab {
		return
	}
	threshold := freqs[ev.MaxVocab]

	newWords := make([]string, 0)
	for _, word := range ev.Words {
		if ev.FreqMap[word] > threshold {
			newWords = append(newWords, word)
		}
	}
	ev.Words = newWords

	newSentences := make([][]string, 0)
	for _, sent := range ev.Sentences {
		newSent := make([]string, 0)
		for _, word := range sent {
			if ev.FreqMap[word] > threshold {
				newSent = append(newSent, word)
			}
		}
		if len(newSent) >= ev.NContext+1 {
			if len(newSent) >= ev.NContext+1 {
				newSentences = append(newSentences, newSent)
			}
		}
	}
	ev.Sentences = newSentences
}

func (ev *CorpusEnv) CreateNGrams() {
	ev.NGrams = make(NGramMap, len(ev.Words))

	for _, sentence := range ev.Sentences {
		size := len(sentence)
		for idx, _ := range sentence {
			if idx >= size-(ev.NContext+1) {
				break
			}
			context := strings.Join(sentence[idx:idx+ev.NContext], " ")
			successor := sentence[idx+ev.NContext]
			ev.NGrams.Add(context, successor)
		}
	}
	ev.NGrams.TopNSuccessors(ev.NSuccessor)
	ev.NGrams.Normalize()
}

func (ev *CorpusEnv) LoadFmFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return err
	}
	defer file.Close()

	byteValue, _ := ioutil.ReadAll(file)
	var data JsonData
	if err := json.Unmarshal(byteValue, &data); err != nil {
		return err
	}
	ev.Words = data.Vocab
	ev.Sentences = data.Sentences
	ev.SentOffs = make([]int, len(ev.Sentences)+1)
	ev.SentOffs[0] = 0
	for i := 1; i <= len(ev.Sentences); i++ {
		ev.SentOffs[i] = ev.SentOffs[i-1] + len(ev.Sentences[i-1])
	}

	ev.WordMap = make(map[string]int)
	ev.FreqMap = make(map[string]float64)
	_, err = os.Stat(ev.VocabFile)
	if os.IsNotExist(err) {
		fmt.Println("Building new vocabulary")

		if ev.UseUNK {
			ev.WordMap["[UNK]"] = 0
			ev.FreqMap["[UNK]"] = 0
		}
		for i, w := range ev.Words {
			ev.WordMap[w] = len(ev.WordMap)
			ev.FreqMap[w] = float64(data.Freqs[i])
		}
		ev.NormalizeFreqs()

		var jobj JsonVocab
		jobj.Vocab = ev.WordMap
		jobj.Freqs = ev.FreqMap
		jenc, _ := json.MarshalIndent(jobj, "", " ")
		_ = ioutil.WriteFile(ev.VocabFile, jenc, 0644)

	} else {
		fmt.Println("Loading existing vocabulary")

		vocab, err := os.Open(ev.VocabFile)
		if err != nil {
			fmt.Println("Error opening vocabulary:", err)
			return err
		}
		defer vocab.Close()

		var jobj JsonVocab
		byteValue, _ := ioutil.ReadAll(vocab)
		if err := json.Unmarshal(byteValue, &jobj); err != nil {
			fmt.Println(err)
			return err
		}
		ev.WordMap = jobj.Vocab
		ev.FreqMap = jobj.Freqs
	}

	// Limit the vocabulary before creating the NGrams, so that NGrams will jump over uncommon words
	ev.LimitVocabulary()
	ev.CreateNGrams()
	return nil
}

// SentToCorpus makes the Corpus out of the Sentences
func (ev *CorpusEnv) SentToCorpus() {
	ev.Corpus = make([]string, 0, len(ev.SentOffs)-1)
	for si := 0; si < len(ev.Sentences); si++ {
		for wi := 0; wi < len(ev.Sentences[si]); wi++ {
			w := ev.LookUpWord(ev.Sentences[si][wi])
			if w != "" {
				ev.Corpus = append(ev.Corpus, w)
			}
		}
	}
}

func (ev *CorpusEnv) ConfigWordReps() {
	nwords := len(ev.Words)
	nin := ev.InputSize.X * ev.InputSize.Y
	nun := int(ev.DistPctAct * float64(nin))
	// TODO is this next line correct?
	nper := nun / (ev.NContext + 1) // each word has this many active, assuming no overlap
	mindif := nper / 2

	ev.WordReps.SetShape([]int{nwords, ev.InputSize.Y, ev.InputSize.X}, nil, []string{"Y", "X"})

	fname := fmt.Sprintf("word_reps_%dx%d_on%d_mind%d_localist%t.json", ev.InputSize.Y, ev.InputSize.X, nper, mindif, ev.Localist)

	_, err := os.Stat(fname)
	if os.IsNotExist(err) {
		fmt.Printf("ConfigWordReps: nwords: %d  nin: %d  nper: %d  minDif: %d\n", nwords, nin, nper, mindif)

		if ev.Localist {
			patgen.MinDiffPrintIters = true
			patgen.PermutedBinaryMinDiff(&ev.WordReps, 1, 1, 0, 1)
			jenc, _ := json.Marshal(ev.WordReps.Values)
			_ = ioutil.WriteFile(fname, jenc, 0644)
		} else {
			patgen.MinDiffPrintIters = true
			patgen.PermutedBinaryMinDiff(&ev.WordReps, 4, 1, 0, mindif)
			jenc, _ := json.Marshal(ev.WordReps.Values)
			_ = ioutil.WriteFile(fname, jenc, 0644)
		}

	} else {
		fmt.Printf("Loading word reps from: %s\n", fname)
		file, err := os.Open(fname)
		if err != nil {
			fmt.Println("Error opening file:", err)
		} else {
			defer file.Close()
			bs, _ := ioutil.ReadAll(file)
			if err := json.Unmarshal(bs, &ev.WordReps.Values); err != nil {
				fmt.Println(err)
			}
		}
	}
}

// CorpusPosToSentIdx returns the sentence, idx of given corpus position
func (ev *CorpusEnv) CorpusPosToSentIdx(pos int) []int {
	idx := make([]int, 2)
	var i, j, curr int
	i = 0
	j = len(ev.SentOffs) - 1
	for {
		curr = (i + j) / 2
		if pos < ev.SentOffs[curr] {
			j = curr - 1
		} else {
			i = curr + 1
		}
		if i >= j {
			idx[0] = curr
			idx[1] = pos - ev.SentOffs[idx[0]]
			break
		}
	}
	return idx
}

// String returns the current state as a string
func (ev *CorpusEnv) String() string {
	curr := make([]string, len(ev.CurWords))
	if ev.CurWords != nil {
		copy(curr, ev.CurWords)
		for i := 0; i < len(curr); i++ {
			if curr[i] == "" {
				curr[i] = "--"
			}
		}
		return strings.Join(curr, " ")
	}
	return ""
}

func (ev *CorpusEnv) AddWordRep(inputoroutput *etensor.Float32, word string) {
	if word == "" {
		return
	}

	widx := ev.WordMap[word]
	wp := ev.WordReps.SubSpace([]int{widx})
	idx := 0
	for y := 0; y < ev.InputSize.Y; y++ {
		for x := 0; x < ev.InputSize.X; x++ {
			wv := wp.FloatVal1D(idx)
			cv := inputoroutput.FloatVal1D(idx)
			nv := math.Max(wv, cv)
			inputoroutput.SetFloat1D(idx, nv)
			idx++
		}
	}
}

// RenderWords renders the current list of CurWords to Input state
func (ev *CorpusEnv) RenderWords() {
	ev.Input.SetZeros()
	for _, word := range ev.CurWords {
		ev.AddWordRep(&ev.Input, word)
	}
	ev.Output.SetZeros()
	ev.AddWordRep(&ev.Output, ev.CurNextWord)
}

func (ev *CorpusEnv) LookUpWord(word string) string {
	ans := word
	if _, ok := ev.WordMap[ans]; !ok {
		if ev.UseUNK {
			ans = "[UNK]"
		} else {
			ans = ""
		}
	} else if ev.DropOut {
		samp := rand.Float64()
		if ev.FreqMap[word]-samp > epsilon {
			ans = ""
		}
	}
	return ans
}

func (ev *CorpusEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.Block.Same() // good idea to just reset all non-inner-most counters at start
	if ev.Tick.Incr() {
		ev.Block.Incr()
	}
	if ev.Trial.Incr() {
		ev.Epoch.Incr()
	}

	// TODO randomly walk words
	// if cur words are empty, pick a random one from the ngrams keys
	// if tick % 100 is 0, pick a random one
	// if the current words do not appear in ngram map, pick a random one
	_, ok := ev.NGrams[strings.Join(ev.CurWords, " ")]
	if len(ev.CurWords) == 0 || ev.Tick.Cur%ev.NRandomizeWord == 0 || !ok {
		// TODO Optimize this
		ridx := rand.Intn(len(ev.NGrams))
		idx := 0
		for context, _ := range ev.NGrams {
			if idx == ridx {
				ev.CurWords = strings.Split(context, " ")
				break
			}
			idx++
		}
		ev.CurNextWord = ""
	}
	// cycle the current words
	if ev.CurNextWord != "" {
		ev.CurWords = append(ev.CurWords[1:], ev.CurNextWord)
		ev.CurNextWord = ""
	}
	// pick a random successor from the ngram map, using weighted random
	possiblechoices := ev.NGrams[strings.Join(ev.CurWords, " ")]
	rando := rand.Float64()
	summo := 0.0
	for word, freq := range possiblechoices {
		summo += freq
		if summo >= rando {
			ev.CurNextWord = word
			break
		}
	}
	if ev.CurNextWord == "" {
		// OH NO ERROR! TODO HOW DO WE ERROR??! :( this should never happen
		return false
	}

	ev.RenderWords()
	return true
}

func (ev *CorpusEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *CorpusEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*CorpusEnv)(nil)
