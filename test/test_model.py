""" Tests for the model builder """

# Imports
import pathlib
import tempfile

from final_project import model, layers

# Tests


def test_create_read_write_model():
    # Make sure we can read and write the model spec

    autoenc = model.Model('3-8-3 Autoencoder', input_size=8)
    autoenc.add_layer(layers.FullyConnected(size=3, func='sigmoid'))
    autoenc.add_layer(layers.FullyConnected(size=8, func='sigmoid'))

    with tempfile.TemporaryDirectory() as tempdir:
        modelfile = pathlib.Path(tempdir) / 'model.json'

        autoenc.save_model(modelfile)
        res_autoenc = model.Model.load_model(modelfile)

        assert res_autoenc == autoenc
