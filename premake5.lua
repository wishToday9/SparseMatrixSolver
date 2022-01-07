
-- Include the premake5 CUDA module
require('premake5-cuda')

workspace "Solver"
    architecture "x64"
    configurations{
        "Debug",
        "Release"
    }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "Solver"
    location "Solver"
    kind "ConsoleApp"
    language "C++"
    --二进制文件
    targetdir ("bin/${outputdir}/%{prj.name}")
    --obj中间文件
    objdir("bin-int/${outputdir}/%{prj.name}")

    files{
        "%{prj.name}/**.h",
        "%{prj.name}/**.txt"
    }

    buildcustomizations "BuildCustomizations/CUDA 11.5"

    cudaFiles {
        "%{prj.name}/Main/*.cu"
    } 

    cudaMaxRegCount "32"  --最大寄存器数量
    
    -- Let's compile for all supported architectures (and also in parallel with -t0)
    cudaCompilerOptions {"-arch=sm_52", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_60,code=sm_60",
                         "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
                         "-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_80,code=sm_80",
                         "-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86", "-t0"}                      
    
    -- On Windows, the link to cudart is done by the CUDA extension, but on Linux, this must be done manually
    if os.target() == "linux" then 
        linkoptions {"-L/usr/local/cuda/lib64 -lcudart"}
    end

    -- includedirs
    -- {
    --     "$(CUDA_PATH)/include"
    -- }
    
    -- libdirs
    -- {
    --     "$(CUDA_PATH)/lib/x64"
    -- }

    -- links
    -- {
    --     "cuda"
    -- }

    filter "system:windows"
        staticruntime "on"
        systemversion "10.0"

    filter "configurations:Debug"
        symbols "On"

    filter "configurations:Release"
        optimize "On"
        cudaFastMath "On" -- enable fast math for release
        

