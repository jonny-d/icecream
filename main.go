package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// basic func to return max value in array
func max(inArray []float64) float64 {

	var maxVal float64 = inArray[0]

	for i := 1; i < len(inArray); i++ {
		if inArray[i] > maxVal {
			maxVal = inArray[i]
		}
	}
	return maxVal
}

type HMM struct {

	// Q is a map for the state space symbols and indices
	Q map[string]int
	// V is a map for observation symbols and indices
	V map[string]int

	A *mat.Dense   // (num_states, num_states) matrix representing transition prob matrix
	B []*mat.Dense // slice of matrices representing emission probabiliy vectors
}

// explicity set all the ice cream example HMM struct fields
func (H *HMM) initIceCreamHMM() {

	// transition probabilities
	a := []float64{0, 0.8, 0.2, 0, 0.7, 0.3, 0, 0.4, 0.6}
	aa := mat.NewDense(3, 3, a)
	H.A = aa

	// cold state emission probabilities
	b0 := []float64{0.2, 0.4, 0.4}
	bb0 := mat.NewDense(1, 3, b0)
	H.B = append(H.B, bb0)

	// hot state emission probabilities
	b1 := []float64{0.5, 0.4, 0.1}
	bb1 := mat.NewDense(1, 3, b1)
	H.B = append(H.B, bb1)

	// state and observations space maps
	H.V = map[string]int{"1": 0, "2": 1, "3": 2}
	H.Q = map[string]int{"START": 0, "HOT": 1, "COLD": 2}

	// matPrint(aa)
	// matPrint(bb0)
	// matPrint(bb1)

}

// given an observation sequence O = o1,o2,...,oT
// and state sequence Q = q1,q1,...,qT
// Calculate the likelihood P(O|Q)
func (H *HMM) likelihood(oSeq, qSeq []string) float64 {
	var p float64 = 1.0
	for i := 0; i < len(oSeq); i++ {
		// H.Q[qSeq[i]]] is gonna be either 0 or 1
		stateIdx := H.Q[qSeq[i]]
		// H.V[oSeq[i]] is gonna be 0, 1 or 2
		emissionIdx := H.V[oSeq[i]]
		p = p * H.B[stateIdx].At(0, emissionIdx)
	}
	return p
}

// given an observation sequence O = o1,o2,...,oT
// calculate the likelihood of the sequence using
// the forward algorithm
func (H *HMM) forward(oSeq []string) float64 {

	// seems complicated but think this is right way to
	// dynamically slice of slices
	table := make([][]float64, len(H.Q)-1)
	table[0] = make([]float64, len(oSeq))
	table[1] = make([]float64, len(oSeq))

	for i := 0; i < len(oSeq); i++ {

		for j := 0; j < len(H.Q)-1; j++ { // -1 because start state doesn't have emission probs

			if i == 0 { // handle the first transition from START state

				trP := H.A.At(0, j+1)     // P(qi|qi-1) transmission probability
				Vidx := H.V[oSeq[i]]      // vocab index of current observation symbol
				emP := H.B[j].At(0, Vidx) // P(oi|qi) emission probability
				fwP := trP * emP          // starting forward path probability
				table[j][i] = fwP

			} else { // transitions between hot and cold states

				Vidx := H.V[oSeq[i]] // vocab index of current observation symbol
				pfwP := 0.0          // var for previous forward path probability
				fwP := 0.0           // var for current forward path probability

				// sum over the incoming paths to each state
				for k := 0; k < len(H.Q)-1; k++ {
					trP := H.A.At(k+1, j+1)   // P(qi|qi-1) transmission probabilities
					emP := H.B[j].At(0, Vidx) // P(oi|qi) emission probabilities
					pfwP = table[k][i-1]
					fwP += trP * emP * pfwP // sum current forward path probability for all paths
				}
				table[j][i] = fwP
			}
		}
	}
	// sum over the probabilities for both of the final states
	p := table[0][len(table[0])-1] + table[1][len(table[1])-1]
	return p
}

// given an observation sequence O = o1,o2,...,oT
// calculate the most likely state sequence
// Q = q1,q2,...,qT using the viterbi algorithm
func (H *HMM) viterbi(oSeq []string) []string {

	table := make([][]float64, len(H.Q)-1)
	table[0] = make([]float64, len(oSeq))
	table[1] = make([]float64, len(oSeq))

	for i := 0; i < len(oSeq); i++ {

		for j := 0; j < len(H.Q)-1; j++ { // -1 because start state doesn't have emission probs

			if i == 0 { // handle the first transition from START state

				trP := H.A.At(0, j+1)     // P(qi|qi-1) transmission probability
				Vidx := H.V[oSeq[i]]      // vocab index of current observation symbol
				emP := H.B[j].At(0, Vidx) // P(oi|qi) emission probability
				fwP := trP * emP          // starting forward path probability
				table[j][i] = fwP

			} else { // transitions between hot and cold states

				Vidx := H.V[oSeq[i]] // vocab index of current observation symbol
				pfwP := 0.0          // var for previous forward path probability
				var fwP []float64    // var for current forward path probability

				// over the incoming paths to each state
				for k := 0; k < len(H.Q)-1; k++ {
					trP := H.A.At(k+1, j+1)   // P(qi|qi-1) transmission probabilities
					emP := H.B[j].At(0, Vidx) // P(oi|qi) emission probabilities
					pfwP = table[k][i-1]
					p := trP * emP * pfwP // sum current forward path probability for all paths
					fwP = append(fwP, p)
				}
				table[j][i] = max(fwP)
			}
		}
	}
	var backtrace []string // return the most likely sequence
	for i := 0; i < len(oSeq); i++ {
		if table[0][i] > table[1][i] {
			backtrace = append(backtrace, "HOT")
		} else {
			backtrace = append(backtrace, "COLD")
		}
	}

	return backtrace
}

func main() {

	var hmm HMM
	hmm.initIceCreamHMM()

	// o := []string{"3", "1", "3"}
	// // q := []string{"HOT", "HOT", "COLD"}

	o := []string{"3", "3", "1", "1", "2", "2", "3", "1", "3"}
	// fmt.Println(hmm.forward(o))
	fmt.Println(hmm.viterbi(o))

	o = []string{"3", "3", "1", "1", "2", "3", "3", "1", "2"}
	// fmt.Println(hmm.forward(o))
	fmt.Println(hmm.viterbi(o))

}
