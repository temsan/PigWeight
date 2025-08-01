from server import create_app, install_requirements, download_model, convert_to_onnx

if __name__ == '__main__':
    install_requirements()
    download_model()
    convert_to_onnx()
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
