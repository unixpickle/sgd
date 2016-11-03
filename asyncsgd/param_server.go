package asyncsgd

import (
	"errors"
	"io/ioutil"
	"net/http"
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	ParamWritePath = "/write_params"
	ParamReadPath  = "/read_params"
)

// A ParamServer coordinates parameters across machines in
// a cluster.
type ParamServer struct {
	paramLock  sync.RWMutex
	parameters []*autofunc.Variable
	updater    Updater
}

// NewParamServer creates a new parameter server.
//
// The parameter server receives gradient updates and
// sends the current parameters to remote entities.
func NewParamServer(params []*autofunc.Variable, u Updater) *ParamServer {
	return &ParamServer{
		parameters: params,
		updater:    u,
	}
}

// ServeHTTP serves the HTTP endpoint for the updater.
//
// The endpoint accepts POSTs to ParamWritePath with a
// set of updates serialized through SerializeUpdates.
// It also accepts GETs to ParamReadPath, from which it
// returns a serialized set of parameters.
func (p *ParamServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/write_params" {
		if err := p.handleWrite(w, r); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
		} else {
			w.Header().Set("Content-Type", "text/plain")
			w.Write([]byte("success"))
		}
	} else if r.URL.Path == "/read_params" {
		p.handleRead(w, r)
	}
}

func (p *ParamServer) handleWrite(w http.ResponseWriter, r *http.Request) error {
	contents, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return err
	}
	vecs, err := DeserializeVectors(contents)
	if err != nil {
		return err
	}
	p.paramLock.Lock()
	defer p.paramLock.Unlock()
	if !p.vecsCompatible(vecs) {
		return errors.New("incompatible vector dimensions")
	}
	grad := autofunc.Gradient{}
	for i, v := range p.parameters {
		grad[v] = vecs[i]
	}
	p.updater.Update(grad)
	return nil
}

func (p *ParamServer) handleRead(w http.ResponseWriter, r *http.Request) {
	p.paramLock.RLock()
	vecs := make([]linalg.Vector, len(p.parameters))
	for i, v := range p.parameters {
		vecs[i] = v.Vector
	}
	enc := SerializeVectors(vecs)
	p.paramLock.RUnlock()
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(enc)
}

func (p *ParamServer) vecsCompatible(v []linalg.Vector) bool {
	if len(v) != len(p.parameters) {
		return false
	}
	for i, x := range p.parameters {
		if len(v[i]) != len(x.Vector) {
			return false
		}
	}
	return true
}
