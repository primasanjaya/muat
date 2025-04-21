nextflow.enable.dsl=2

workflow {
    runMuatPredict(
        file(params.input_vcf),
        file(params.reference),
        params.result_dir
    )
}

process runMuatPredict {
    // Leave out conda/container config here — controlled via profile
    input:
    path input_vcf
    path reference
    val result_dir

    output:
    path result_dir

    script:
    def input_name = input_vcf.getName()
    def ref_name = reference.getName()
    """
    muat predict wgs \\
        --hg19 ./${ref_name} \\
        --mutation-type '${params.mutation_type}' \\
        --input-filepath ./${input_name} \\
        --result-dir ${result_dir}
    """
}
