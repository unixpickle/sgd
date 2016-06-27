package sgd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
)

type equilibrationTestGradienter struct {
	Var *autofunc.Variable
}

func (e equilibrationTestGradienter) Gradient(s SampleSet) autofunc.Gradient {
	constVec := &autofunc.Variable{Vector: []float64{1, 3}}
	sum := autofunc.SumAll(autofunc.Mul(autofunc.Mul(e.Var, e.Var), constVec))
	grad := autofunc.NewGradient(e.Parameters())
	sum.PropagateGradient([]float64{1}, grad)
	return grad
}

func (e equilibrationTestGradienter) RGradient(rv autofunc.RVector,
	s SampleSet) (autofunc.Gradient, autofunc.RGradient) {
	constVec := &autofunc.Variable{Vector: []float64{1, 3}}
	constRVec := autofunc.NewRVariable(constVec, rv)
	rVar := autofunc.NewRVariable(e.Var, rv)
	sum := autofunc.SumAllR(autofunc.MulR(autofunc.MulR(rVar, rVar), constRVec))
	grad := autofunc.NewGradient(e.Parameters())
	rgrad := autofunc.NewRGradient(e.Parameters())
	sum.PropagateRGradient([]float64{1}, []float64{0}, rgrad, grad)
	return grad, rgrad
}

func (e equilibrationTestGradienter) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{e.Var}
}

func TestEquilibrationManySamples(t *testing.T) {
	rand.Seed(123)
	gradienter := equilibrationTestGradienter{
		Var: &autofunc.Variable{Vector: []float64{0.5, -0.8}},
	}
	eq := Equilibration{
		RGradienter: gradienter,
		Learner:     gradienter,
		NumSamples:  1000,
	}
	grad := eq.Gradient(nil)
	actual := grad[gradienter.Var]
	expected := []float64{2 * 0.5 / 2, 2 * 3 * -0.8 / 6}

	for i, x := range expected {
		a := actual[i]
		if math.Abs(x-a) > 5e-2 {
			t.Errorf("index %d: expected %f got %f", i, x, a)
		}
	}
}
