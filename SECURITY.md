# Security Policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub: **Security → Report a vulnerability** on this repository.

Do not open a public issue or PR for:

- vulnerabilities in `pasteur-cli`, the crates, or the Python bindings;
- anything that could expose patient data or PHI, such as simulation output leaking source rows it should not, or unsafe handling of model or dataset files.

We aim to acknowledge reports within 3 business days and will keep you updated until a fix ships.

## Supported versions

Security fixes land on the latest released minor version.
