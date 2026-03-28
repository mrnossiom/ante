use std::process::Command;

pub mod llvm;

pub fn link_with_gcc(object_filename: &str, binary_filename: &str) -> bool {
    // call gcc to compile the bitcode to a binary
    let output = format!("-o{}", binary_filename);
    let mut child = Command::new("gcc")
        .arg(object_filename)
        //.arg(minicoro_path())
        .arg("-Wno-everything")
        .arg("-O0")
        .arg("-lm")
        .arg(output)
        .spawn()
        .unwrap();

    // remove the temporary bitcode file
    let status = child.wait().unwrap();
    std::fs::remove_file(object_filename).unwrap();
    status.success()
}
