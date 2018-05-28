Name:           charm
Version:        6.8
Release:        2
Summary:        Message-passing parallel language and runtime system
License:        Non-Exclusive, Non-Commercial
Source:         https://charm.cs.illinois.edu/distrib/charm-%{version}.%{release}.tar.gz
URL:            https://hpccharm.com
Vendor:         Charmworks <info@hpccharm.com>

%description
Charm++ is a message-passing parallel language and runtime system. It is implemented as a set of libraries for C++, is efficient, and is portable to a wide variety of parallel machines. Source code is provided, and non-commercial use is free.

%license
LICENSE

%prep
%setup -q -n charm-v%{version}.%{release}

%build
./build LIBS multicore-linux-x86_64 --with-production

%clean
rm -rf %{buildroot}

%install
mkdir -p %{buildroot}/usr/local
cd tmp; make DESTDIR=%{buildroot}/usr/local/%{name} install

%files
/usr/local/charm/

%doc README CHANGES README.ampi README.bigsim
