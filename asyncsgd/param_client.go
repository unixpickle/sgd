package asyncsgd

import (
	"bytes"
	"errors"
	"io/ioutil"
	"net/http"
	"net/url"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A ParamClient interacts with a ParamServer.
type ParamClient struct {
	// BaseURL is the URL of the root directory on the
	// parameter server.
	// The proper endpoints will be added on to this URL.
	BaseURL *url.URL
}

// ReadParams requests the parameters from the server and
// writes them into the variables.
func (p *ParamClient) ReadParams(out []*autofunc.Variable) error {
	u := *p.BaseURL
	u.Path = ParamReadPath
	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if resp != nil {
		defer resp.Body.Close()
	}
	if err != nil {
		return err
	}
	contents, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	vecs, err := DeserializeVectors(contents)
	if err != nil {
		return err
	}
	if len(vecs) != len(out) {
		return errors.New("incompatible dimensions")
	}
	for i, x := range vecs {
		if len(out[i].Vector) != len(x) {
			return errors.New("incompatible dimensions")
		}
	}
	for i, x := range vecs {
		copy(out[i].Vector, x)
	}
	return nil
}

// WriteParams sends a parameter update.
func (p *ParamClient) WriteParams(g autofunc.Gradient, v []*autofunc.Variable) error {
	updateVecs := make([]linalg.Vector, len(v))
	for i, variable := range v {
		updateVecs[i] = g[variable]
	}
	encoded := SerializeVectors(updateVecs)

	u := *p.BaseURL
	u.Path = ParamWritePath
	req, err := http.NewRequest("POST", u.String(), bytes.NewReader(encoded))
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if resp.Body != nil {
		defer resp.Body.Close()
	}
	if err != nil {
		return err
	}
	if resp.StatusCode != 200 {
		errStr, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return err
		} else {
			return errors.New("remote error: " + string(errStr))
		}
	}
	return nil
}
