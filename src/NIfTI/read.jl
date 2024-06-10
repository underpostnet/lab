

# Read: Neuroimaging Informatics Technology Initiative (NIfTI)

using NIfTI

ni = niread("../../data/Neurohacking_data-0.0/BRAINIX/NIfTI/BRAINIX_NIFTI_Output_3D_File.nii.gz") # gzipped NIfTI files are detected automatically


println(size(ni))

println(ni.header)

# println(ni.raw)